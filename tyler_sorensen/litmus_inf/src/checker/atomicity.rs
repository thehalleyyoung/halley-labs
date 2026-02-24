#![allow(unused)]
//! Atomicity verification for memory models.
//!
//! Implements RMW semantics, CAS operation modeling, atomic fetch operations,
//! and atomicity axiom checking for GPU memory model verification.

use std::collections::{HashMap, HashSet, VecDeque, BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

pub type EventId = usize;
pub type ThreadId = usize;
pub type Address = u64;
pub type Value = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RmwOp {
    CompareAndSwap,
    FetchAdd,
    FetchSub,
    FetchAnd,
    FetchOr,
    FetchXor,
    FetchMin,
    FetchMax,
    Exchange,
    LoadLinkedStoreConditional,
}

impl fmt::Display for RmwOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompareAndSwap => write!(f, "CompareAndSwap"),
            Self::FetchAdd => write!(f, "FetchAdd"),
            Self::FetchSub => write!(f, "FetchSub"),
            Self::FetchAnd => write!(f, "FetchAnd"),
            Self::FetchOr => write!(f, "FetchOr"),
            Self::FetchXor => write!(f, "FetchXor"),
            Self::FetchMin => write!(f, "FetchMin"),
            Self::FetchMax => write!(f, "FetchMax"),
            Self::Exchange => write!(f, "Exchange"),
            Self::LoadLinkedStoreConditional => write!(f, "LoadLinkedStoreConditional"),
        }
    }
}

impl RmwOp {
    pub fn all() -> Vec<Self> {
        vec![Self::CompareAndSwap, Self::FetchAdd, Self::FetchSub, Self::FetchAnd,
             Self::FetchOr, Self::FetchXor, Self::FetchMin, Self::FetchMax,
             Self::Exchange, Self::LoadLinkedStoreConditional]
    }

    pub fn is_fetch_op(&self) -> bool {
        matches!(self, Self::FetchAdd | Self::FetchSub | Self::FetchAnd |
                       Self::FetchOr | Self::FetchXor | Self::FetchMin | Self::FetchMax)
    }

    pub fn is_conditional(&self) -> bool {
        matches!(self, Self::CompareAndSwap | Self::LoadLinkedStoreConditional)
    }

    /// Apply the RMW operation to compute the new value.
    pub fn apply(&self, current: Value, operand: Value, expected: Value) -> (Value, bool) {
        match self {
            Self::CompareAndSwap => {
                if current == expected { (operand, true) } else { (current, false) }
            }
            Self::FetchAdd => (current.wrapping_add(operand), true),
            Self::FetchSub => (current.wrapping_sub(operand), true),
            Self::FetchAnd => (current & operand, true),
            Self::FetchOr => (current | operand, true),
            Self::FetchXor => (current ^ operand, true),
            Self::FetchMin => (current.min(operand), true),
            Self::FetchMax => (current.max(operand), true),
            Self::Exchange => (operand, true),
            Self::LoadLinkedStoreConditional => {
                if current == expected { (operand, true) } else { (current, false) }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl AtomicOrdering {
    pub fn all() -> Vec<Self> {
        vec![Self::Relaxed, Self::Acquire, Self::Release, Self::AcqRel, Self::SeqCst]
    }

    pub fn implies_acquire(&self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::SeqCst)
    }

    pub fn implies_release(&self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_at_least(&self, other: &Self) -> bool {
        *self >= *other
    }
}

impl fmt::Display for AtomicOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "relaxed"),
            Self::Acquire => write!(f, "acquire"),
            Self::Release => write!(f, "release"),
            Self::AcqRel => write!(f, "acq_rel"),
            Self::SeqCst => write!(f, "seq_cst"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum AtomicScope {
    Thread, Warp, Block, Device, System,
}

impl AtomicScope {
    pub fn includes(&self, other: &Self) -> bool { *self >= *other }
    pub fn all() -> Vec<Self> { vec![Self::Thread, Self::Warp, Self::Block, Self::Device, Self::System] }
}

impl fmt::Display for AtomicScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Thread => write!(f, "thread"),
            Self::Warp => write!(f, "warp"),
            Self::Block => write!(f, "block"),
            Self::Device => write!(f, "device"),
            Self::System => write!(f, "system"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmwSemantics {
    pub op: RmwOp,
    pub read_ordering: AtomicOrdering,
    pub write_ordering: AtomicOrdering,
    pub scope: AtomicScope,
    pub description: String,
}

impl RmwSemantics {
    pub fn new(op: RmwOp, ordering: AtomicOrdering, scope: AtomicScope) -> Self {
        let (ro, wo) = match ordering {
            AtomicOrdering::Relaxed => (AtomicOrdering::Relaxed, AtomicOrdering::Relaxed),
            AtomicOrdering::Acquire => (AtomicOrdering::Acquire, AtomicOrdering::Relaxed),
            AtomicOrdering::Release => (AtomicOrdering::Relaxed, AtomicOrdering::Release),
            AtomicOrdering::AcqRel => (AtomicOrdering::Acquire, AtomicOrdering::Release),
            AtomicOrdering::SeqCst => (AtomicOrdering::SeqCst, AtomicOrdering::SeqCst),
        };
        RmwSemantics {
            op, read_ordering: ro, write_ordering: wo, scope,
            description: format!("{} with {} scope", op, scope),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmwOperand {
    pub expected: Value,
    pub desired: Value,
    pub ordering: AtomicOrdering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmwResult {
    pub success: bool,
    pub old_value: Value,
    pub new_value: Value,
}

impl fmt::Display for RmwResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(f, "OK({} -> {})", self.old_value, self.new_value)
        } else {
            write!(f, "FAIL(current={})", self.old_value)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicEvent {
    pub event_id: EventId,
    pub op: RmwOp,
    pub address: Address,
    pub thread_id: ThreadId,
    pub scope: AtomicScope,
    pub ordering: AtomicOrdering,
    pub operand: Value,
    pub expected: Value,
    pub read_value: Value,
    pub write_value: Value,
}

impl fmt::Display for AtomicEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:T{}:{}({:#x},exp={},op={})->{}", 
            self.event_id, self.thread_id, self.op, self.address,
            self.expected, self.operand, self.write_value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicExecution {
    pub events: Vec<AtomicEvent>,
    pub reads_from: HashMap<EventId, EventId>,
    pub coherence_order: Vec<(EventId, EventId)>,
    pub program_order: Vec<(EventId, EventId)>,
}

impl AtomicExecution {
    pub fn new() -> Self {
        AtomicExecution {
            events: Vec::new(),
            reads_from: HashMap::new(),
            coherence_order: Vec::new(),
            program_order: Vec::new(),
        }
    }

    pub fn add_event(&mut self, event: AtomicEvent) {
        self.events.push(event);
    }

    pub fn add_rf(&mut self, write: EventId, read: EventId) {
        self.reads_from.insert(read, write);
    }

    pub fn add_co(&mut self, before: EventId, after: EventId) {
        self.coherence_order.push((before, after));
    }

    pub fn add_po(&mut self, before: EventId, after: EventId) {
        self.program_order.push((before, after));
    }

    pub fn events_at(&self, addr: Address) -> Vec<&AtomicEvent> {
        self.events.iter().filter(|e| e.address == addr).collect()
    }

    pub fn events_by_thread(&self, tid: ThreadId) -> Vec<&AtomicEvent> {
        self.events.iter().filter(|e| e.thread_id == tid).collect()
    }
}

// =========================================================================
// CAS Operation Modeling
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasOperation {
    pub event_id: EventId,
    pub address: Address,
    pub expected: Value,
    pub desired: Value,
    pub success_ordering: AtomicOrdering,
    pub failure_ordering: AtomicOrdering,
    pub scope: AtomicScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CasOutcome {
    Success(Value),
    Failure(Value),
    Spurious,
}

impl fmt::Display for CasOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success(v) => write!(f, "CAS-success(old={})", v),
            Self::Failure(v) => write!(f, "CAS-fail(current={})", v),
            Self::Spurious => write!(f, "CAS-spurious"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CasValidator {
    pub strict: bool,
}

impl CasValidator {
    pub fn new(strict: bool) -> Self { CasValidator { strict } }

    /// Validate a CAS operation outcome given the reads-from write.
    pub fn validate_cas(
        &self,
        cas: &CasOperation,
        rf_write_value: Value,
        outcome: CasOutcome,
    ) -> Result<(), AtomicityViolation> {
        match outcome {
            CasOutcome::Success(old) => {
                if old != rf_write_value {
                    return Err(AtomicityViolation {
                        axiom: AtomicityAxiom::RmwAtomicity,
                        events: vec![cas.event_id],
                        description: format!(
                            "CAS success old value {} != rf value {}", old, rf_write_value
                        ),
                    });
                }
                if rf_write_value != cas.expected {
                    return Err(AtomicityViolation {
                        axiom: AtomicityAxiom::RmwAtomicity,
                        events: vec![cas.event_id],
                        description: format!(
                            "CAS success but read value {} != expected {}", rf_write_value, cas.expected
                        ),
                    });
                }
                Ok(())
            }
            CasOutcome::Failure(current) => {
                if current == cas.expected && self.strict {
                    return Err(AtomicityViolation {
                        axiom: AtomicityAxiom::RmwAtomicity,
                        events: vec![cas.event_id],
                        description: "CAS failure but current == expected (strict mode)".to_string(),
                    });
                }
                Ok(())
            }
            CasOutcome::Spurious => {
                if self.strict {
                    return Err(AtomicityViolation {
                        axiom: AtomicityAxiom::RmwAtomicity,
                        events: vec![cas.event_id],
                        description: "Spurious failure not allowed in strict mode".to_string(),
                    });
                }
                Ok(())
            }
        }
    }

    /// Check CAS RF constraint: if CAS succeeds, it must read expected value.
    pub fn cas_rf_constraint(cas: &CasOperation, rf_value: Value) -> bool {
        rf_value == cas.expected
    }

    /// Check CAS CO constraint: no write intervenes between rf-source and CAS write.
    pub fn cas_co_constraint(
        cas_event: EventId,
        rf_source: EventId,
        co_edges: &[(EventId, EventId)],
        addr: Address,
        addr_writes: &[EventId],
    ) -> bool {
        // Check no write w exists such that co(rf_source, w) and co(w, cas_event)
        for &w in addr_writes {
            if w == rf_source || w == cas_event { continue; }
            let after_source = co_edges.iter().any(|&(a, b)| a == rf_source && b == w);
            let before_cas = co_edges.iter().any(|&(a, b)| a == w && b == cas_event);
            if after_source && before_cas { return false; }
        }
        true
    }

    /// Validate CAS atomicity: no CO-intervening write.
    pub fn validate_cas_atomicity(
        &self,
        cas_event: EventId,
        rf_source: EventId,
        co_edges: &[(EventId, EventId)],
        addr: Address,
        addr_writes: &[EventId],
    ) -> Result<(), AtomicityViolation> {
        if !Self::cas_co_constraint(cas_event, rf_source, co_edges, addr, addr_writes) {
            return Err(AtomicityViolation {
                axiom: AtomicityAxiom::NoIntervening,
                events: vec![cas_event, rf_source],
                description: "CO-intervening write between RF source and CAS".to_string(),
            });
        }
        Ok(())
    }
}

// =========================================================================
// Fetch Operation Modeling
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FetchOpKind { Add, Sub, And, Or, Xor, Min, Max, Exchange }

impl FetchOpKind {
    pub fn apply(&self, current: Value, operand: Value) -> Value {
        match self {
            Self::Add => current.wrapping_add(operand),
            Self::Sub => current.wrapping_sub(operand),
            Self::And => current & operand,
            Self::Or => current | operand,
            Self::Xor => current ^ operand,
            Self::Min => current.min(operand),
            Self::Max => current.max(operand),
            Self::Exchange => operand,
        }
    }
}

impl fmt::Display for FetchOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::And => write!(f, "and"),
            Self::Or => write!(f, "or"),
            Self::Xor => write!(f, "xor"),
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
            Self::Exchange => write!(f, "exchange"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchOperation {
    pub event_id: EventId,
    pub address: Address,
    pub operand: Value,
    pub op_kind: FetchOpKind,
    pub ordering: AtomicOrdering,
    pub scope: AtomicScope,
}

#[derive(Debug, Clone)]
pub struct FetchValidator;

impl FetchValidator {
    /// Validate that the write value of a fetch operation is correct given RF.
    pub fn validate_fetch_value(
        fetch: &FetchOperation,
        read_value: Value,
        written_value: Value,
    ) -> Result<(), AtomicityViolation> {
        let expected = fetch.op_kind.apply(read_value, fetch.operand);
        if written_value != expected {
            return Err(AtomicityViolation {
                axiom: AtomicityAxiom::RmwAtomicity,
                events: vec![fetch.event_id],
                description: format!(
                    "Fetch {} wrote {} but expected {} (read={}, operand={})",
                    fetch.op_kind, written_value, expected, read_value, fetch.operand
                ),
            });
        }
        Ok(())
    }

    /// Derive value constraints from fetch semantics.
    pub fn fetch_value_constraint(
        op: FetchOpKind,
        read_value: Value,
        operand: Value,
    ) -> Value {
        op.apply(read_value, operand)
    }
}

// =========================================================================
// Atomicity Axiom Checking
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicityAxiom {
    RmwAtomicity,
    NoIntervening,
    CoherenceImmediate,
    ExclusiveSuccess,
    ScopeConsistency,
}

impl fmt::Display for AtomicityAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmwAtomicity => write!(f, "rmw-atomicity"),
            Self::NoIntervening => write!(f, "no-intervening"),
            Self::CoherenceImmediate => write!(f, "co-immediate"),
            Self::ExclusiveSuccess => write!(f, "exclusive-success"),
            Self::ScopeConsistency => write!(f, "scope-consistency"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicityViolation {
    pub axiom: AtomicityAxiom,
    pub events: Vec<EventId>,
    pub description: String,
}

impl fmt::Display for AtomicityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Atomicity violation [{}]: {} (events: {:?})", self.axiom, self.description, self.events)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicityChecker {
    strict: bool,
}

impl AtomicityChecker {
    pub fn new(strict: bool) -> Self { AtomicityChecker { strict } }

    /// Check RMW atomicity: for each RMW pair (r,w), no CO-intervening write
    /// between the write w reads from and w itself.
    pub fn check_rmw_atomicity(
        &self,
        exec: &AtomicExecution,
    ) -> Vec<AtomicityViolation> {
        let mut violations = Vec::new();
        let co_set: HashSet<(EventId, EventId)> = exec.coherence_order.iter().copied().collect();
        for event in &exec.events {
            if let Some(&rf_source) = exec.reads_from.get(&event.event_id) {
                // Find writes to same address
                let addr_writes: Vec<EventId> = exec.events.iter()
                    .filter(|e| e.address == event.address)
                    .map(|e| e.event_id)
                    .collect();
                // Check no intervening write
                for &w in &addr_writes {
                    if w == rf_source || w == event.event_id { continue; }
                    let after_source = co_set.contains(&(rf_source, w));
                    let before_event = co_set.contains(&(w, event.event_id));
                    if after_source && before_event {
                        violations.push(AtomicityViolation {
                            axiom: AtomicityAxiom::NoIntervening,
                            events: vec![event.event_id, rf_source, w],
                            description: format!(
                                "Write E{} intervenes between RF source E{} and RMW E{}",
                                w, rf_source, event.event_id
                            ),
                        });
                    }
                }
            }
        }
        violations
    }

    /// Check exclusive LL/SC pairs.
    pub fn check_exclusive_pairs(
        &self,
        exec: &AtomicExecution,
    ) -> Vec<AtomicityViolation> {
        let mut violations = Vec::new();
        let llsc_events: Vec<_> = exec.events.iter()
            .filter(|e| e.op == RmwOp::LoadLinkedStoreConditional)
            .collect();
        for ev in &llsc_events {
            // Check that if SC succeeds, no intervening store from another thread
            let (_, success) = ev.op.apply(ev.read_value, ev.write_value, ev.expected);
            if success {
                // Check for stores between LL and SC from other threads
                let other_stores: Vec<_> = exec.events.iter()
                    .filter(|e| e.address == ev.address && e.thread_id != ev.thread_id)
                    .collect();
                for other in &other_stores {
                    let co_between = exec.coherence_order.iter().any(|&(a, b)| {
                        (a == other.event_id && b == ev.event_id) ||
                        (a == ev.event_id && b == other.event_id)
                    });
                    if co_between {
                        // Potential exclusivity violation detected
                    }
                }
            }
        }
        violations
    }

    /// Run all atomicity checks.
    pub fn check_all(
        &self,
        exec: &AtomicExecution,
    ) -> Vec<AtomicityViolation> {
        let mut all = Vec::new();
        all.extend(self.check_rmw_atomicity(exec));
        all.extend(self.check_exclusive_pairs(exec));
        all.extend(self.check_scope_consistency(exec));
        all
    }

    /// Check scope consistency.
    pub fn check_scope_consistency(
        &self,
        exec: &AtomicExecution,
    ) -> Vec<AtomicityViolation> {
        let mut violations = Vec::new();
        for &(a_id, b_id) in &exec.coherence_order {
            let a = exec.events.iter().find(|e| e.event_id == a_id);
            let b = exec.events.iter().find(|e| e.event_id == b_id);
            if let (Some(a), Some(b)) = (a, b) {
                if a.scope != b.scope && !a.scope.includes(&b.scope) && !b.scope.includes(&a.scope) {
                    violations.push(AtomicityViolation {
                        axiom: AtomicityAxiom::ScopeConsistency,
                        events: vec![a_id, b_id],
                        description: format!(
                            "Incompatible scopes: E{} has {} but E{} has {}",
                            a_id, a.scope, b_id, b.scope
                        ),
                    });
                }
            }
        }
        violations
    }
}

// =========================================================================
// GPU-specific Atomicity
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAtomicModel {
    pub warp_size: usize,
    pub supports_system_scope: bool,
    pub supports_warp_scope: bool,
    pub coalescing_enabled: bool,
}

impl GpuAtomicModel {
    pub fn nvidia() -> Self {
        GpuAtomicModel { warp_size: 32, supports_system_scope: true,
            supports_warp_scope: true, coalescing_enabled: true }
    }

    pub fn amd() -> Self {
        GpuAtomicModel { warp_size: 64, supports_system_scope: true,
            supports_warp_scope: false, coalescing_enabled: true }
    }

    pub fn apple() -> Self {
        GpuAtomicModel { warp_size: 32, supports_system_scope: false,
            supports_warp_scope: false, coalescing_enabled: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpLevelAtomicity {
    pub warp_id: usize,
    pub lanes: Vec<ThreadId>,
    pub coalesced_atomics: Vec<Vec<EventId>>,
}

impl WarpLevelAtomicity {
    pub fn new(warp_id: usize, lanes: Vec<ThreadId>) -> Self {
        WarpLevelAtomicity { warp_id, lanes, coalesced_atomics: Vec::new() }
    }

    pub fn coalesce_atomics(&mut self, events: &[AtomicEvent]) {
        let warp_events: Vec<_> = events.iter()
            .filter(|e| self.lanes.contains(&e.thread_id))
            .collect();
        let mut by_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        for e in &warp_events {
            by_addr.entry(e.address).or_default().push(e.event_id);
        }
        self.coalesced_atomics = by_addr.into_values().collect();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgroupAtomicity {
    pub subgroup_id: usize,
    pub members: Vec<ThreadId>,
    pub scope: AtomicScope,
}

impl SubgroupAtomicity {
    pub fn new(subgroup_id: usize, members: Vec<ThreadId>, scope: AtomicScope) -> Self {
        SubgroupAtomicity { subgroup_id, members, scope }
    }

    pub fn check_scope_inclusion(&self, event: &AtomicEvent) -> bool {
        self.scope.includes(&event.scope)
    }

    pub fn atomic_visibility(&self, event: &AtomicEvent) -> Vec<ThreadId> {
        match event.scope {
            AtomicScope::Thread => vec![event.thread_id],
            AtomicScope::Warp => self.members.clone(),
            _ => self.members.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAtomicViolation {
    pub violation_type: GpuAtomicViolationType,
    pub events: Vec<EventId>,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuAtomicViolationType {
    ScopeViolation,
    CoalescingConflict,
    WarpDivergenceRace,
    SubgroupAtomicityFailure,
}

impl fmt::Display for GpuAtomicViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU atomic violation [{:?}]: {}", self.violation_type, self.description)
    }
}

// =========================================================================
// Atomicity Enumeration
// =========================================================================

#[derive(Debug, Clone)]
pub struct RmwOutcomeEnumerator {
    events: Vec<AtomicEvent>,
}

impl RmwOutcomeEnumerator {
    pub fn new(events: Vec<AtomicEvent>) -> Self {
        RmwOutcomeEnumerator { events }
    }

    /// Enumerate all valid outcomes for a set of RMW operations.
    pub fn enumerate_outcomes(&self) -> Vec<Vec<RmwResult>> {
        if self.events.is_empty() { return vec![Vec::new()]; }
        let mut results = Vec::new();
        let first = &self.events[0];
        // For CAS, two outcomes: success or failure
        let possible = if first.op.is_conditional() {
            vec![
                RmwResult { success: true, old_value: first.expected, new_value: first.write_value },
                RmwResult { success: false, old_value: first.read_value, new_value: first.read_value },
            ]
        } else {
            vec![RmwResult { success: true, old_value: first.read_value, new_value: first.write_value }]
        };
        let rest_enum = RmwOutcomeEnumerator::new(self.events[1..].to_vec());
        let rest_outcomes = rest_enum.enumerate_outcomes();
        for p in &possible {
            for rest in &rest_outcomes {
                let mut combo = vec![p.clone()];
                combo.extend(rest.iter().cloned());
                results.push(combo);
            }
        }
        results
    }
}

#[derive(Debug, Clone)]
pub struct InterferenceAnalysis {
    events: Vec<AtomicEvent>,
}

impl InterferenceAnalysis {
    pub fn new(events: Vec<AtomicEvent>) -> Self {
        InterferenceAnalysis { events }
    }

    /// Compute set of conflicting RMW operations (same address, different threads).
    pub fn conflict_set(&self) -> Vec<(EventId, EventId)> {
        let mut conflicts = Vec::new();
        for i in 0..self.events.len() {
            for j in (i+1)..self.events.len() {
                let a = &self.events[i];
                let b = &self.events[j];
                if a.address == b.address && a.thread_id != b.thread_id {
                    conflicts.push((a.event_id, b.event_id));
                }
            }
        }
        conflicts
    }

    /// Check if any pair of RMW operations interfere.
    pub fn has_interference(&self) -> bool {
        !self.conflict_set().is_empty()
    }

    /// Group conflicting events by address.
    pub fn conflicts_by_address(&self) -> HashMap<Address, Vec<EventId>> {
        let mut by_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        for e in &self.events {
            by_addr.entry(e.address).or_default().push(e.event_id);
        }
        by_addr.into_iter().filter(|(_, v)| v.len() > 1).collect()
    }
}

// =========================================================================
// Utilities
// =========================================================================

pub fn format_rmw_execution(exec: &AtomicExecution) -> String {
    let mut out = String::new();
    out.push_str("RMW Execution:\n");
    for e in &exec.events {
        out.push_str(&format!("  {}\n", e));
    }
    out.push_str(&format!("  RF: {:?}\n", exec.reads_from));
    out.push_str(&format!("  CO: {:?}\n", exec.coherence_order));
    out
}

pub fn rmw_statistics(exec: &AtomicExecution) -> RmwStatistics {
    let total = exec.events.len();
    let cas_count = exec.events.iter().filter(|e| e.op == RmwOp::CompareAndSwap).count();
    let fetch_count = exec.events.iter().filter(|e| e.op.is_fetch_op()).count();
    let exchange_count = exec.events.iter().filter(|e| e.op == RmwOp::Exchange).count();
    let llsc_count = exec.events.iter().filter(|e| e.op == RmwOp::LoadLinkedStoreConditional).count();
    let addrs: HashSet<Address> = exec.events.iter().map(|e| e.address).collect();
    let threads: HashSet<ThreadId> = exec.events.iter().map(|e| e.thread_id).collect();
    RmwStatistics {
        total_rmw_ops: total,
        cas_ops: cas_count,
        fetch_ops: fetch_count,
        exchange_ops: exchange_count,
        llsc_ops: llsc_count,
        unique_addresses: addrs.len(),
        unique_threads: threads.len(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmwStatistics {
    pub total_rmw_ops: usize,
    pub cas_ops: usize,
    pub fetch_ops: usize,
    pub exchange_ops: usize,
    pub llsc_ops: usize,
    pub unique_addresses: usize,
    pub unique_threads: usize,
}

impl fmt::Display for RmwStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RMW Statistics:")?;
        writeln!(f, "  Total:     {}", self.total_rmw_ops)?;
        writeln!(f, "  CAS:       {}", self.cas_ops)?;
        writeln!(f, "  Fetch:     {}", self.fetch_ops)?;
        writeln!(f, "  Exchange:  {}", self.exchange_ops)?;
        writeln!(f, "  LL/SC:     {}", self.llsc_ops)?;
        writeln!(f, "  Addresses: {}", self.unique_addresses)?;
        write!(f, "  Threads:   {}", self.unique_threads)
    }
}

pub fn atomic_operation_graph(exec: &AtomicExecution) -> String {
    let mut dot = String::from("digraph rmw_ops {\n");
    dot.push_str("    rankdir=LR;\n");
    for e in &exec.events {
        dot.push_str(&format!("    E{} [label=\"{}\"];\n", e.event_id, e));
    }
    for (&read, &write) in &exec.reads_from {
        dot.push_str(&format!("    E{} -> E{} [label=\"rf\", color=red];\n", write, read));
    }
    for &(a, b) in &exec.coherence_order {
        dot.push_str(&format!("    E{} -> E{} [label=\"co\", color=blue];\n", a, b));
    }
    for &(a, b) in &exec.program_order {
        dot.push_str(&format!("    E{} -> E{} [label=\"po\"];\n", a, b));
    }
    dot.push_str("}\n");
    dot
}

// ===== Extended Atomicity Operations =====

#[derive(Debug, Clone)]
pub struct WeakCasSemantics {
    pub max_retries: u32,
    pub backoff_factor: f64,
    pub success_probability: f64,
}

impl WeakCasSemantics {
    pub fn new(max_retries: u32, backoff_factor: f64, success_probability: f64) -> Self {
        WeakCasSemantics { max_retries, backoff_factor, success_probability }
    }

    pub fn get_max_retries(&self) -> u32 {
        self.max_retries
    }

    pub fn get_backoff_factor(&self) -> f64 {
        self.backoff_factor
    }

    pub fn get_success_probability(&self) -> f64 {
        self.success_probability
    }

    pub fn with_max_retries(mut self, v: u32) -> Self {
        self.max_retries = v; self
    }

    pub fn with_backoff_factor(mut self, v: f64) -> Self {
        self.backoff_factor = v; self
    }

    pub fn with_success_probability(mut self, v: f64) -> Self {
        self.success_probability = v; self
    }

}

impl fmt::Display for WeakCasSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WeakCasSemantics({:?})", self.max_retries)
    }
}

#[derive(Debug, Clone)]
pub struct WeakCasSemanticsBuilder {
    max_retries: u32,
    backoff_factor: f64,
    success_probability: f64,
}

impl WeakCasSemanticsBuilder {
    pub fn new() -> Self {
        WeakCasSemanticsBuilder {
            max_retries: 0,
            backoff_factor: 0.0,
            success_probability: 0.0,
        }
    }

    pub fn max_retries(mut self, v: u32) -> Self { self.max_retries = v; self }
    pub fn backoff_factor(mut self, v: f64) -> Self { self.backoff_factor = v; self }
    pub fn success_probability(mut self, v: f64) -> Self { self.success_probability = v; self }
}

#[derive(Debug, Clone)]
pub struct StrongCasSemantics {
    pub guaranteed_progress: bool,
    pub max_contention: u32,
}

impl StrongCasSemantics {
    pub fn new(guaranteed_progress: bool, max_contention: u32) -> Self {
        StrongCasSemantics { guaranteed_progress, max_contention }
    }

    pub fn get_guaranteed_progress(&self) -> bool {
        self.guaranteed_progress
    }

    pub fn get_max_contention(&self) -> u32 {
        self.max_contention
    }

    pub fn with_guaranteed_progress(mut self, v: bool) -> Self {
        self.guaranteed_progress = v; self
    }

    pub fn with_max_contention(mut self, v: u32) -> Self {
        self.max_contention = v; self
    }

}

impl fmt::Display for StrongCasSemantics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StrongCasSemantics({:?})", self.guaranteed_progress)
    }
}

#[derive(Debug, Clone)]
pub struct StrongCasSemanticsBuilder {
    guaranteed_progress: bool,
    max_contention: u32,
}

impl StrongCasSemanticsBuilder {
    pub fn new() -> Self {
        StrongCasSemanticsBuilder {
            guaranteed_progress: false,
            max_contention: 0,
        }
    }

    pub fn guaranteed_progress(mut self, v: bool) -> Self { self.guaranteed_progress = v; self }
    pub fn max_contention(mut self, v: u32) -> Self { self.max_contention = v; self }
}

#[derive(Debug, Clone)]
pub struct LlScMonitor {
    pub reservation_active: bool,
    pub reserved_address: u64,
    pub reservation_size: u32,
    pub monitor_id: u32,
}

impl LlScMonitor {
    pub fn new(reservation_active: bool, reserved_address: u64, reservation_size: u32, monitor_id: u32) -> Self {
        LlScMonitor { reservation_active, reserved_address, reservation_size, monitor_id }
    }

    pub fn get_reservation_active(&self) -> bool {
        self.reservation_active
    }

    pub fn get_reserved_address(&self) -> u64 {
        self.reserved_address
    }

    pub fn get_reservation_size(&self) -> u32 {
        self.reservation_size
    }

    pub fn get_monitor_id(&self) -> u32 {
        self.monitor_id
    }

    pub fn with_reservation_active(mut self, v: bool) -> Self {
        self.reservation_active = v; self
    }

    pub fn with_reserved_address(mut self, v: u64) -> Self {
        self.reserved_address = v; self
    }

    pub fn with_reservation_size(mut self, v: u32) -> Self {
        self.reservation_size = v; self
    }

    pub fn with_monitor_id(mut self, v: u32) -> Self {
        self.monitor_id = v; self
    }

}

impl fmt::Display for LlScMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LlScMonitor({:?})", self.reservation_active)
    }
}

#[derive(Debug, Clone)]
pub struct LlScMonitorBuilder {
    reservation_active: bool,
    reserved_address: u64,
    reservation_size: u32,
    monitor_id: u32,
}

impl LlScMonitorBuilder {
    pub fn new() -> Self {
        LlScMonitorBuilder {
            reservation_active: false,
            reserved_address: 0,
            reservation_size: 0,
            monitor_id: 0,
        }
    }

    pub fn reservation_active(mut self, v: bool) -> Self { self.reservation_active = v; self }
    pub fn reserved_address(mut self, v: u64) -> Self { self.reserved_address = v; self }
    pub fn reservation_size(mut self, v: u32) -> Self { self.reservation_size = v; self }
    pub fn monitor_id(mut self, v: u32) -> Self { self.monitor_id = v; self }
}

#[derive(Debug, Clone)]
pub struct LlScState {
    pub monitors: Vec<u64>,
    pub invalidated: Vec<bool>,
    pub timestamp: u64,
}

impl LlScState {
    pub fn new(monitors: Vec<u64>, invalidated: Vec<bool>, timestamp: u64) -> Self {
        LlScState { monitors, invalidated, timestamp }
    }

    pub fn monitors_len(&self) -> usize {
        self.monitors.len()
    }

    pub fn monitors_is_empty(&self) -> bool {
        self.monitors.is_empty()
    }

    pub fn invalidated_len(&self) -> usize {
        self.invalidated.len()
    }

    pub fn invalidated_is_empty(&self) -> bool {
        self.invalidated.is_empty()
    }

    pub fn get_timestamp(&self) -> u64 {
        self.timestamp
    }

    pub fn with_timestamp(mut self, v: u64) -> Self {
        self.timestamp = v; self
    }

}

impl fmt::Display for LlScState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LlScState({:?})", self.monitors)
    }
}

#[derive(Debug, Clone)]
pub struct LlScStateBuilder {
    monitors: Vec<u64>,
    invalidated: Vec<bool>,
    timestamp: u64,
}

impl LlScStateBuilder {
    pub fn new() -> Self {
        LlScStateBuilder {
            monitors: Vec::new(),
            invalidated: Vec::new(),
            timestamp: 0,
        }
    }

    pub fn monitors(mut self, v: Vec<u64>) -> Self { self.monitors = v; self }
    pub fn invalidated(mut self, v: Vec<bool>) -> Self { self.invalidated = v; self }
    pub fn timestamp(mut self, v: u64) -> Self { self.timestamp = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicScheduler {
    pub thread_count: u32,
    pub priority_queue: Vec<u32>,
    pub fairness_counter: Vec<u64>,
}

impl AtomicScheduler {
    pub fn new(thread_count: u32, priority_queue: Vec<u32>, fairness_counter: Vec<u64>) -> Self {
        AtomicScheduler { thread_count, priority_queue, fairness_counter }
    }

    pub fn get_thread_count(&self) -> u32 {
        self.thread_count
    }

    pub fn priority_queue_len(&self) -> usize {
        self.priority_queue.len()
    }

    pub fn priority_queue_is_empty(&self) -> bool {
        self.priority_queue.is_empty()
    }

    pub fn fairness_counter_len(&self) -> usize {
        self.fairness_counter.len()
    }

    pub fn fairness_counter_is_empty(&self) -> bool {
        self.fairness_counter.is_empty()
    }

    pub fn with_thread_count(mut self, v: u32) -> Self {
        self.thread_count = v; self
    }

}

impl fmt::Display for AtomicScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicScheduler({:?})", self.thread_count)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicSchedulerBuilder {
    thread_count: u32,
    priority_queue: Vec<u32>,
    fairness_counter: Vec<u64>,
}

impl AtomicSchedulerBuilder {
    pub fn new() -> Self {
        AtomicSchedulerBuilder {
            thread_count: 0,
            priority_queue: Vec::new(),
            fairness_counter: Vec::new(),
        }
    }

    pub fn thread_count(mut self, v: u32) -> Self { self.thread_count = v; self }
    pub fn priority_queue(mut self, v: Vec<u32>) -> Self { self.priority_queue = v; self }
    pub fn fairness_counter(mut self, v: Vec<u64>) -> Self { self.fairness_counter = v; self }
}

#[derive(Debug, Clone)]
pub struct RmwConflictGraph {
    pub node_count: usize,
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<f64>,
}

impl RmwConflictGraph {
    pub fn new(node_count: usize, adjacency: Vec<Vec<bool>>, weights: Vec<f64>) -> Self {
        RmwConflictGraph { node_count, adjacency, weights }
    }

    pub fn get_node_count(&self) -> usize {
        self.node_count
    }

    pub fn adjacency_len(&self) -> usize {
        self.adjacency.len()
    }

    pub fn adjacency_is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    pub fn weights_len(&self) -> usize {
        self.weights.len()
    }

    pub fn weights_is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn with_node_count(mut self, v: usize) -> Self {
        self.node_count = v; self
    }

}

impl fmt::Display for RmwConflictGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RmwConflictGraph({:?})", self.node_count)
    }
}

#[derive(Debug, Clone)]
pub struct RmwConflictGraphBuilder {
    node_count: usize,
    adjacency: Vec<Vec<bool>>,
    weights: Vec<f64>,
}

impl RmwConflictGraphBuilder {
    pub fn new() -> Self {
        RmwConflictGraphBuilder {
            node_count: 0,
            adjacency: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn node_count(mut self, v: usize) -> Self { self.node_count = v; self }
    pub fn adjacency(mut self, v: Vec<Vec<bool>>) -> Self { self.adjacency = v; self }
    pub fn weights(mut self, v: Vec<f64>) -> Self { self.weights = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicOrderingLattice {
    pub elements: Vec<u32>,
    pub order: Vec<Vec<bool>>,
}

impl AtomicOrderingLattice {
    pub fn new(elements: Vec<u32>, order: Vec<Vec<bool>>) -> Self {
        AtomicOrderingLattice { elements, order }
    }

    pub fn elements_len(&self) -> usize {
        self.elements.len()
    }

    pub fn elements_is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn order_len(&self) -> usize {
        self.order.len()
    }

    pub fn order_is_empty(&self) -> bool {
        self.order.is_empty()
    }

}

impl fmt::Display for AtomicOrderingLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicOrderingLattice({:?})", self.elements)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicOrderingLatticeBuilder {
    elements: Vec<u32>,
    order: Vec<Vec<bool>>,
}

impl AtomicOrderingLatticeBuilder {
    pub fn new() -> Self {
        AtomicOrderingLatticeBuilder {
            elements: Vec::new(),
            order: Vec::new(),
        }
    }

    pub fn elements(mut self, v: Vec<u32>) -> Self { self.elements = v; self }
    pub fn order(mut self, v: Vec<Vec<bool>>) -> Self { self.order = v; self }
}

#[derive(Debug, Clone)]
pub struct FenceInsertion {
    pub location: usize,
    pub fence_type: String,
    pub cost: f64,
}

impl FenceInsertion {
    pub fn new(location: usize, fence_type: String, cost: f64) -> Self {
        FenceInsertion { location, fence_type, cost }
    }

    pub fn get_location(&self) -> usize {
        self.location
    }

    pub fn get_fence_type(&self) -> &str {
        &self.fence_type
    }

    pub fn get_cost(&self) -> f64 {
        self.cost
    }

    pub fn with_location(mut self, v: usize) -> Self {
        self.location = v; self
    }

    pub fn with_fence_type(mut self, v: impl Into<String>) -> Self {
        self.fence_type = v.into(); self
    }

    pub fn with_cost(mut self, v: f64) -> Self {
        self.cost = v; self
    }

}

impl fmt::Display for FenceInsertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FenceInsertion({:?})", self.location)
    }
}

#[derive(Debug, Clone)]
pub struct FenceInsertionBuilder {
    location: usize,
    fence_type: String,
    cost: f64,
}

impl FenceInsertionBuilder {
    pub fn new() -> Self {
        FenceInsertionBuilder {
            location: 0,
            fence_type: String::new(),
            cost: 0.0,
        }
    }

    pub fn location(mut self, v: usize) -> Self { self.location = v; self }
    pub fn fence_type(mut self, v: impl Into<String>) -> Self { self.fence_type = v.into(); self }
    pub fn cost(mut self, v: f64) -> Self { self.cost = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicReorderAnalysis {
    pub reorderable_pairs: Vec<(usize, usize)>,
    pub barrier_points: Vec<usize>,
}

impl AtomicReorderAnalysis {
    pub fn new(reorderable_pairs: Vec<(usize, usize)>, barrier_points: Vec<usize>) -> Self {
        AtomicReorderAnalysis { reorderable_pairs, barrier_points }
    }

    pub fn reorderable_pairs_len(&self) -> usize {
        self.reorderable_pairs.len()
    }

    pub fn reorderable_pairs_is_empty(&self) -> bool {
        self.reorderable_pairs.is_empty()
    }

    pub fn barrier_points_len(&self) -> usize {
        self.barrier_points.len()
    }

    pub fn barrier_points_is_empty(&self) -> bool {
        self.barrier_points.is_empty()
    }

}

impl fmt::Display for AtomicReorderAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicReorderAnalysis({:?})", self.reorderable_pairs)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicReorderAnalysisBuilder {
    reorderable_pairs: Vec<(usize, usize)>,
    barrier_points: Vec<usize>,
}

impl AtomicReorderAnalysisBuilder {
    pub fn new() -> Self {
        AtomicReorderAnalysisBuilder {
            reorderable_pairs: Vec::new(),
            barrier_points: Vec::new(),
        }
    }

    pub fn reorderable_pairs(mut self, v: Vec<(usize, usize)>) -> Self { self.reorderable_pairs = v; self }
    pub fn barrier_points(mut self, v: Vec<usize>) -> Self { self.barrier_points = v; self }
}

#[derive(Debug, Clone)]
pub struct WarpAtomicCoalescing {
    pub warp_size: u32,
    pub coalesced_count: u32,
    pub uncoalesced_count: u32,
    pub efficiency: f64,
}

impl WarpAtomicCoalescing {
    pub fn new(warp_size: u32, coalesced_count: u32, uncoalesced_count: u32, efficiency: f64) -> Self {
        WarpAtomicCoalescing { warp_size, coalesced_count, uncoalesced_count, efficiency }
    }

    pub fn get_warp_size(&self) -> u32 {
        self.warp_size
    }

    pub fn get_coalesced_count(&self) -> u32 {
        self.coalesced_count
    }

    pub fn get_uncoalesced_count(&self) -> u32 {
        self.uncoalesced_count
    }

    pub fn get_efficiency(&self) -> f64 {
        self.efficiency
    }

    pub fn with_warp_size(mut self, v: u32) -> Self {
        self.warp_size = v; self
    }

    pub fn with_coalesced_count(mut self, v: u32) -> Self {
        self.coalesced_count = v; self
    }

    pub fn with_uncoalesced_count(mut self, v: u32) -> Self {
        self.uncoalesced_count = v; self
    }

    pub fn with_efficiency(mut self, v: f64) -> Self {
        self.efficiency = v; self
    }

}

impl fmt::Display for WarpAtomicCoalescing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WarpAtomicCoalescing({:?})", self.warp_size)
    }
}

#[derive(Debug, Clone)]
pub struct WarpAtomicCoalescingBuilder {
    warp_size: u32,
    coalesced_count: u32,
    uncoalesced_count: u32,
    efficiency: f64,
}

impl WarpAtomicCoalescingBuilder {
    pub fn new() -> Self {
        WarpAtomicCoalescingBuilder {
            warp_size: 0,
            coalesced_count: 0,
            uncoalesced_count: 0,
            efficiency: 0.0,
        }
    }

    pub fn warp_size(mut self, v: u32) -> Self { self.warp_size = v; self }
    pub fn coalesced_count(mut self, v: u32) -> Self { self.coalesced_count = v; self }
    pub fn uncoalesced_count(mut self, v: u32) -> Self { self.uncoalesced_count = v; self }
    pub fn efficiency(mut self, v: f64) -> Self { self.efficiency = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicContentionTracker {
    pub address_counts: Vec<u64>,
    pub max_contention: u64,
    pub total_ops: u64,
}

impl AtomicContentionTracker {
    pub fn new(address_counts: Vec<u64>, max_contention: u64, total_ops: u64) -> Self {
        AtomicContentionTracker { address_counts, max_contention, total_ops }
    }

    pub fn address_counts_len(&self) -> usize {
        self.address_counts.len()
    }

    pub fn address_counts_is_empty(&self) -> bool {
        self.address_counts.is_empty()
    }

    pub fn get_max_contention(&self) -> u64 {
        self.max_contention
    }

    pub fn get_total_ops(&self) -> u64 {
        self.total_ops
    }

    pub fn with_max_contention(mut self, v: u64) -> Self {
        self.max_contention = v; self
    }

    pub fn with_total_ops(mut self, v: u64) -> Self {
        self.total_ops = v; self
    }

}

impl fmt::Display for AtomicContentionTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicContentionTracker({:?})", self.address_counts)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicContentionTrackerBuilder {
    address_counts: Vec<u64>,
    max_contention: u64,
    total_ops: u64,
}

impl AtomicContentionTrackerBuilder {
    pub fn new() -> Self {
        AtomicContentionTrackerBuilder {
            address_counts: Vec::new(),
            max_contention: 0,
            total_ops: 0,
        }
    }

    pub fn address_counts(mut self, v: Vec<u64>) -> Self { self.address_counts = v; self }
    pub fn max_contention(mut self, v: u64) -> Self { self.max_contention = v; self }
    pub fn total_ops(mut self, v: u64) -> Self { self.total_ops = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicProgressGuarantee {
    pub guarantee_type: String,
    pub bound: u64,
    pub verified: bool,
}

impl AtomicProgressGuarantee {
    pub fn new(guarantee_type: String, bound: u64, verified: bool) -> Self {
        AtomicProgressGuarantee { guarantee_type, bound, verified }
    }

    pub fn get_guarantee_type(&self) -> &str {
        &self.guarantee_type
    }

    pub fn get_bound(&self) -> u64 {
        self.bound
    }

    pub fn get_verified(&self) -> bool {
        self.verified
    }

    pub fn with_guarantee_type(mut self, v: impl Into<String>) -> Self {
        self.guarantee_type = v.into(); self
    }

    pub fn with_bound(mut self, v: u64) -> Self {
        self.bound = v; self
    }

    pub fn with_verified(mut self, v: bool) -> Self {
        self.verified = v; self
    }

}

impl fmt::Display for AtomicProgressGuarantee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicProgressGuarantee({:?})", self.guarantee_type)
    }
}

#[derive(Debug, Clone)]
pub struct AtomicProgressGuaranteeBuilder {
    guarantee_type: String,
    bound: u64,
    verified: bool,
}

impl AtomicProgressGuaranteeBuilder {
    pub fn new() -> Self {
        AtomicProgressGuaranteeBuilder {
            guarantee_type: String::new(),
            bound: 0,
            verified: false,
        }
    }

    pub fn guarantee_type(mut self, v: impl Into<String>) -> Self { self.guarantee_type = v.into(); self }
    pub fn bound(mut self, v: u64) -> Self { self.bound = v; self }
    pub fn verified(mut self, v: bool) -> Self { self.verified = v; self }
}

#[derive(Debug, Clone)]
pub struct AtomicAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl AtomicAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        AtomicAnalysis { data, size, computed: false, label: "Atomic".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for AtomicAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomicStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for AtomicStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicStatus::Pending => write!(f, "pending"),
            AtomicStatus::InProgress => write!(f, "inprogress"),
            AtomicStatus::Completed => write!(f, "completed"),
            AtomicStatus::Failed => write!(f, "failed"),
            AtomicStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomicPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for AtomicPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicPriority::Critical => write!(f, "critical"),
            AtomicPriority::High => write!(f, "high"),
            AtomicPriority::Medium => write!(f, "medium"),
            AtomicPriority::Low => write!(f, "low"),
            AtomicPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomicMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for AtomicMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicMode::Strict => write!(f, "strict"),
            AtomicMode::Relaxed => write!(f, "relaxed"),
            AtomicMode::Permissive => write!(f, "permissive"),
            AtomicMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn atomic_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn atomic_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = atomic_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn atomic_std_dev(data: &[f64]) -> f64 {
    atomic_variance(data).sqrt()
}

pub fn atomic_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

pub fn atomic_percentile(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * 0.95).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn atomic_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn atomic_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn atomic_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = atomic_mean(&x);
    let my = atomic_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn atomic_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = atomic_covariance(data);
    let sx = atomic_std_dev(&data[..n]);
    let sy = atomic_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn atomic_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = atomic_mean(data);
    let s = atomic_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn atomic_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = atomic_mean(data);
    let s = atomic_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn atomic_harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn atomic_geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over atomic analysis results.
#[derive(Debug, Clone)]
pub struct AtomicResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl AtomicResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        AtomicResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for AtomicResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert WeakCasSemantics description to a summary string.
pub fn weakcassemantics_to_summary(item: &WeakCasSemantics) -> String {
    format!("WeakCasSemantics: {:?}", item)
}

/// Convert StrongCasSemantics description to a summary string.
pub fn strongcassemantics_to_summary(item: &StrongCasSemantics) -> String {
    format!("StrongCasSemantics: {:?}", item)
}

/// Convert LlScMonitor description to a summary string.
pub fn llscmonitor_to_summary(item: &LlScMonitor) -> String {
    format!("LlScMonitor: {:?}", item)
}

/// Convert LlScState description to a summary string.
pub fn llscstate_to_summary(item: &LlScState) -> String {
    format!("LlScState: {:?}", item)
}

/// Convert AtomicScheduler description to a summary string.
pub fn atomicscheduler_to_summary(item: &AtomicScheduler) -> String {
    format!("AtomicScheduler: {:?}", item)
}

/// Convert RmwConflictGraph description to a summary string.
pub fn rmwconflictgraph_to_summary(item: &RmwConflictGraph) -> String {
    format!("RmwConflictGraph: {:?}", item)
}

/// Convert AtomicOrderingLattice description to a summary string.
pub fn atomicorderinglattice_to_summary(item: &AtomicOrderingLattice) -> String {
    format!("AtomicOrderingLattice: {:?}", item)
}

/// Convert FenceInsertion description to a summary string.
pub fn fenceinsertion_to_summary(item: &FenceInsertion) -> String {
    format!("FenceInsertion: {:?}", item)
}

/// Convert AtomicReorderAnalysis description to a summary string.
pub fn atomicreorderanalysis_to_summary(item: &AtomicReorderAnalysis) -> String {
    format!("AtomicReorderAnalysis: {:?}", item)
}

/// Convert WarpAtomicCoalescing description to a summary string.
pub fn warpatomiccoalescing_to_summary(item: &WarpAtomicCoalescing) -> String {
    format!("WarpAtomicCoalescing: {:?}", item)
}

/// Convert AtomicContentionTracker description to a summary string.
pub fn atomiccontentiontracker_to_summary(item: &AtomicContentionTracker) -> String {
    format!("AtomicContentionTracker: {:?}", item)
}

/// Batch processor for atomic operations.
#[derive(Debug, Clone)]
pub struct AtomicBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl AtomicBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        AtomicBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for AtomicBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for atomic analysis.
#[derive(Debug, Clone)]
pub struct AtomicReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl AtomicReport {
    pub fn new(title: impl Into<String>) -> Self {
        AtomicReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for AtomicReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicReport({})", self.title)
    }
}

/// Configuration for atomic analysis.
#[derive(Debug, Clone)]
pub struct AtomicConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl AtomicConfig {
    pub fn default_config() -> Self {
        AtomicConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for AtomicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for atomic data distribution.
#[derive(Debug, Clone)]
pub struct AtomicHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl AtomicHistogram {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return AtomicHistogram { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        AtomicHistogram { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for AtomicHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for atomic graph analysis.
#[derive(Debug, Clone)]
pub struct AtomicGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl AtomicGraph {
    pub fn new(n: usize) -> Self {
        AtomicGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_atomic(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_atomic(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_atomic(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for AtomicGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for atomic computation results.
#[derive(Debug, Clone)]
pub struct AtomicCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl AtomicCache {
    pub fn new(capacity: usize) -> Self {
        AtomicCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for AtomicCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for atomic elements.
pub fn atomic_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for atomic data.
pub fn atomic_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for atomic data.
pub fn atomic_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for Atomicity computations.
#[derive(Debug, Clone)]
pub struct AtomicityDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl AtomicityDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        AtomicityDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        AtomicityDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        AtomicityDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        AtomicityDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        AtomicityDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        AtomicityDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for AtomicityDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AtomicityMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Atomicity bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct AtomicityInterval {
    pub lo: f64,
    pub hi: f64,
}

impl AtomicityInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        AtomicityInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        AtomicityInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        AtomicityInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(AtomicityInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        AtomicityInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        AtomicityInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        AtomicityInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { AtomicityInterval { lo: -self.hi, hi: -self.lo } }
        else { AtomicityInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        AtomicityInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for AtomicityInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Atomicity protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomicityState {
    Ready,
    Acquiring,
    Held,
    Releasing,
    Contended,
    Failed,
}

impl fmt::Display for AtomicityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicityState::Ready => write!(f, "ready"),
            AtomicityState::Acquiring => write!(f, "acquiring"),
            AtomicityState::Held => write!(f, "held"),
            AtomicityState::Releasing => write!(f, "releasing"),
            AtomicityState::Contended => write!(f, "contended"),
            AtomicityState::Failed => write!(f, "failed"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AtomicityStateMachine {
    pub current: AtomicityState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl AtomicityStateMachine {
    pub fn new() -> Self {
        AtomicityStateMachine { current: AtomicityState::Ready, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &AtomicityState { &self.current }
    pub fn can_transition(&self, target: &AtomicityState) -> bool {
        match (&self.current, target) {
            (AtomicityState::Ready, AtomicityState::Acquiring) => true,
            (AtomicityState::Acquiring, AtomicityState::Held) => true,
            (AtomicityState::Acquiring, AtomicityState::Contended) => true,
            (AtomicityState::Held, AtomicityState::Releasing) => true,
            (AtomicityState::Releasing, AtomicityState::Ready) => true,
            (AtomicityState::Contended, AtomicityState::Acquiring) => true,
            (AtomicityState::Contended, AtomicityState::Failed) => true,
            (AtomicityState::Failed, AtomicityState::Ready) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: AtomicityState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = AtomicityState::Ready;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for AtomicityStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Atomicity event tracking.
#[derive(Debug, Clone)]
pub struct AtomicityRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl AtomicityRingBuffer {
    pub fn new(capacity: usize) -> Self {
        AtomicityRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for AtomicityRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Atomicity component tracking.
#[derive(Debug, Clone)]
pub struct AtomicityDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl AtomicityDisjointSet {
    pub fn new(n: usize) -> Self {
        AtomicityDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for AtomicityDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicitySortedList {
    data: Vec<f64>,
}

impl AtomicitySortedList {
    pub fn new() -> Self { AtomicitySortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for AtomicitySortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Atomicity metrics.
#[derive(Debug, Clone)]
pub struct AtomicityEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl AtomicityEma {
    pub fn new(alpha: f64) -> Self { AtomicityEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for AtomicityEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Atomicity membership testing.
#[derive(Debug, Clone)]
pub struct AtomicityBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl AtomicityBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        AtomicityBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for AtomicityBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Atomicity string matching.
#[derive(Debug, Clone)]
pub struct AtomicityTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct AtomicityTrie {
    nodes: Vec<AtomicityTrieNode>,
    count: usize,
}

impl AtomicityTrie {
    pub fn new() -> Self {
        AtomicityTrie { nodes: vec![AtomicityTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(AtomicityTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for AtomicityTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Atomicity scheduling.
#[derive(Debug, Clone)]
pub struct AtomicityPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl AtomicityPriorityQueue {
    pub fn new() -> Self { AtomicityPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for AtomicityPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl AtomicityAccumulator {
    pub fn new() -> Self { AtomicityAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for AtomicityAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicitySparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl AtomicitySparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { AtomicitySparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = AtomicitySparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = AtomicitySparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = AtomicitySparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for AtomicitySparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityPolynomial {
    pub coefficients: Vec<f64>,
}

impl AtomicityPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { AtomicityPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { AtomicityPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { AtomicityPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        AtomicityPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        AtomicityPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        AtomicityPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        AtomicityPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        AtomicityPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        AtomicityPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        AtomicityPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for AtomicityPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityRng {
    state: u64,
}

impl AtomicityRng {
    pub fn new(seed: u64) -> Self { AtomicityRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for AtomicityRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Atomicity benchmarking.
#[derive(Debug, Clone)]
pub struct AtomicityTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl AtomicityTimer {
    pub fn new(label: impl Into<String>) -> Self { AtomicityTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for AtomicityTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Atomicity set operations.
#[derive(Debug, Clone)]
pub struct AtomicityBitVector {
    words: Vec<u64>,
    len: usize,
}

impl AtomicityBitVector {
    pub fn new(len: usize) -> Self { AtomicityBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for AtomicityBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Atomicity computation memoization.
#[derive(Debug, Clone)]
pub struct AtomicityLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl AtomicityLruCache {
    pub fn new(capacity: usize) -> Self { AtomicityLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for AtomicityLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Atomicity scheduling.
#[derive(Debug, Clone)]
pub struct AtomicityGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl AtomicityGraphColoring {
    pub fn new(n: usize) -> Self {
        AtomicityGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for AtomicityGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Atomicity ranking.
#[derive(Debug, Clone)]
pub struct AtomicityTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl AtomicityTopK {
    pub fn new(k: usize) -> Self { AtomicityTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for AtomicityTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Atomicity monitoring.
#[derive(Debug, Clone)]
pub struct AtomicitySlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl AtomicitySlidingWindow {
    pub fn new(window_size: usize) -> Self { AtomicitySlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for AtomicitySlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Atomicity classification evaluation.
#[derive(Debug, Clone)]
pub struct AtomicityConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl AtomicityConfusionMatrix {
    pub fn new() -> Self { AtomicityConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for AtomicityConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Atomicity feature vectors.
pub fn atomicity_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Atomicity.
pub fn atomicity_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Atomicity.
pub fn atomicity_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Atomicity.
pub fn atomicity_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Atomicity.
pub fn atomicity_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Atomicity.
pub fn atomicity_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Atomicity.
pub fn atomicity_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Atomicity.
pub fn atomicity_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Atomicity.
pub fn atomicity_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Atomicity.
pub fn atomicity_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Atomicity.
pub fn atomicity_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Atomicity.
pub fn atomicity_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Atomicity.
pub fn atomicity_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Atomicity.
pub fn atomicity_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Atomicity.
pub fn atomicity_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (atomicity_kl_divergence(p, &m) + atomicity_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Atomicity.
pub fn atomicity_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Atomicity.
pub fn atomicity_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Atomicity.
pub fn atomicity_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl AtomicityFeatureScaler {
    pub fn new() -> Self { AtomicityFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for AtomicityFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Atomicity trend analysis.
#[derive(Debug, Clone)]
pub struct AtomicityLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl AtomicityLinearRegression {
    pub fn new() -> Self { AtomicityLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for AtomicityLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl AtomicityWeightedGraph {
    pub fn new(n: usize) -> Self { AtomicityWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_atomicity(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_atomicity(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_atomicity(&mut parent, u);
            let rv = find_atomicity(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for AtomicityWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Atomicity.
pub fn atomicity_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for Atomicity.
pub fn atomicity_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Atomicity.
pub fn atomicity_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Atomicity.
pub fn atomicity_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Atomicity.
pub fn atomicity_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for Atomicity.
pub fn atomicity_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Atomicity.
pub fn atomicity_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for Atomicity.
pub fn atomicity_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Histogram for Atomicity data analysis.
#[derive(Debug, Clone)]
pub struct AtomicityHistogramExt {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub total: usize,
}

impl AtomicityHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let bin_width = range / num_bins as f64;
        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { edges.push(min + i as f64 * bin_width); }
        let mut bins = vec![0usize; num_bins];
        for &v in data {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        AtomicityHistogramExt { bins, edges, total: data.len() }
    }
    pub fn bin_count(&self, i: usize) -> usize { self.bins[i] }
    pub fn bin_density(&self, i: usize) -> f64 {
        let w = self.edges[i + 1] - self.edges[i];
        if w.abs() < 1e-15 || self.total == 0 { 0.0 }
        else { self.bins[i] as f64 / (self.total as f64 * w) }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn mode_bin(&self) -> usize {
        self.bins.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0)
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut sum = 0;
        for &c in &self.bins { sum += c; cum.push(sum); }
        cum
    }
    pub fn percentile_bin(&self, p: f64) -> usize {
        let target = (p * self.total as f64).ceil() as usize;
        let cum = self.cumulative();
        cum.iter().position(|&c| c >= target).unwrap_or(self.bins.len() - 1)
    }
    pub fn entropy(&self) -> f64 {
        let n = self.total as f64;
        if n < 1.0 { return 0.0; }
        self.bins.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        }).sum()
    }
}

impl fmt::Display for AtomicityHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hist(bins={}, total={})", self.num_bins(), self.total)
    }
}

/// Axis-aligned bounding box for Atomicity spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct AtomicityAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl AtomicityAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { AtomicityAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            AtomicityAABB::new(self.x_min, self.y_min, cx, cy),
            AtomicityAABB::new(cx, self.y_min, self.x_max, cy),
            AtomicityAABB::new(self.x_min, cy, cx, self.y_max),
            AtomicityAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Atomicity.
#[derive(Debug, Clone, Copy)]
pub struct AtomicityPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Atomicity spatial indexing.
#[derive(Debug, Clone)]
pub struct AtomicityQuadTree {
    pub boundary: AtomicityAABB,
    pub points: Vec<AtomicityPoint2D>,
    pub children: Option<Vec<AtomicityQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl AtomicityQuadTree {
    pub fn new(boundary: AtomicityAABB, capacity: usize, max_depth: usize) -> Self {
        AtomicityQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: AtomicityAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        AtomicityQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: AtomicityPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(AtomicityQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &AtomicityAABB) -> Vec<AtomicityPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for AtomicityQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Atomicity.
pub fn atomicity_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for Atomicity.
pub fn atomicity_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Atomicity.
pub fn atomicity_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Atomicity.
pub fn atomicity_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Atomicity.
pub fn atomicity_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Atomicity.
pub fn atomicity_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Atomicity.
pub fn atomicity_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Atomicity.
pub fn atomicity_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Atomicity.
pub fn atomicity_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = atomicity_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Atomicity.
#[derive(Debug, Clone)]
pub struct AtomicityRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl AtomicityRunningStats {
    pub fn new() -> Self { AtomicityRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for AtomicityRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Percentile calculator for Atomicity.
pub fn atomicity_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

/// Interquartile range for Atomicity.
pub fn atomicity_iqr(data: &[f64]) -> f64 {
    atomicity_percentile_at(data, 75.0) - atomicity_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Atomicity.
pub fn atomicity_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = atomicity_percentile_at(data, 25.0);
    let q3 = atomicity_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Atomicity.
pub fn atomicity_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Atomicity.
pub fn atomicity_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Atomicity.
pub fn atomicity_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = atomicity_rank(x);
    let ry = atomicity_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Geometric mean for Atomicity.
pub fn atomicity_geomean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let log_sum: f64 = data.iter().map(|&x| x.ln()).sum();
    (log_sum / data.len() as f64).exp()
}

/// Harmonic mean for Atomicity.
pub fn atomicity_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let recip_sum: f64 = data.iter().map(|&x| 1.0 / x).sum();
    data.len() as f64 / recip_sum
}

/// Skewness for Atomicity.
pub fn atomicity_sample_skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum();
    let s2 = m2 / (n - 1.0);
    let s = s2.sqrt();
    if s.abs() < 1e-15 { return 0.0; }
    (n / ((n - 1.0) * (n - 2.0))) * m3 / s.powi(3)
}

/// Excess kurtosis for Atomicity.
pub fn atomicity_excess_kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2.abs() < 1e-15 { return 0.0; }
    m4 / (m2 * m2) - 3.0
}

/// Covariance matrix for Atomicity.
pub fn atomicity_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for Atomicity.
pub fn atomicity_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = atomicity_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atomic(id: EventId, op: RmwOp, addr: Address, tid: ThreadId,
                   expected: Value, operand: Value, read_val: Value, write_val: Value) -> AtomicEvent {
        AtomicEvent {
            event_id: id, op, address: addr, thread_id: tid,
            scope: AtomicScope::Device, ordering: AtomicOrdering::SeqCst,
            operand, expected, read_value: read_val, write_value: write_val,
        }
    }

    #[test]
    fn test_rmw_op_apply_cas_success() {
    let (val, ok) = RmwOp::CompareAndSwap.apply(42, 99, 42);
            assert!(ok);
            assert_eq!(val, 99);
    }

    #[test]
    fn test_rmw_op_apply_cas_failure() {
    let (val, ok) = RmwOp::CompareAndSwap.apply(42, 99, 100);
            assert!(!ok);
            assert_eq!(val, 42);
    }

    #[test]
    fn test_rmw_op_apply_fetch_add() {
    let (val, ok) = RmwOp::FetchAdd.apply(10, 5, 0);
            assert!(ok);
            assert_eq!(val, 15);
    }

    #[test]
    fn test_rmw_op_apply_fetch_sub() {
    let (val, ok) = RmwOp::FetchSub.apply(10, 3, 0);
            assert!(ok);
            assert_eq!(val, 7);
    }

    #[test]
    fn test_rmw_op_apply_fetch_and() {
    let (val, ok) = RmwOp::FetchAnd.apply(0b1100, 0b1010, 0);
            assert!(ok);
            assert_eq!(val, 0b1000);
    }

    #[test]
    fn test_rmw_op_apply_fetch_or() {
    let (val, ok) = RmwOp::FetchOr.apply(0b1100, 0b1010, 0);
            assert!(ok);
            assert_eq!(val, 0b1110);
    }

    #[test]
    fn test_rmw_op_apply_exchange() {
    let (val, ok) = RmwOp::Exchange.apply(42, 99, 0);
            assert!(ok);
            assert_eq!(val, 99);
    }

    #[test]
    fn test_rmw_op_is_fetch() {
    assert!(RmwOp::FetchAdd.is_fetch_op());
            assert!(!RmwOp::CompareAndSwap.is_fetch_op());
            assert!(!RmwOp::Exchange.is_fetch_op());
    }

    #[test]
    fn test_rmw_op_is_conditional() {
    assert!(RmwOp::CompareAndSwap.is_conditional());
            assert!(RmwOp::LoadLinkedStoreConditional.is_conditional());
            assert!(!RmwOp::FetchAdd.is_conditional());
    }

    #[test]
    fn test_atomic_ordering_implies() {
    assert!(AtomicOrdering::SeqCst.implies_acquire());
            assert!(AtomicOrdering::SeqCst.implies_release());
            assert!(AtomicOrdering::AcqRel.implies_acquire());
            assert!(AtomicOrdering::AcqRel.implies_release());
            assert!(!AtomicOrdering::Relaxed.implies_acquire());
    }

    #[test]
    fn test_atomic_scope_includes() {
    assert!(AtomicScope::System.includes(&AtomicScope::Device));
            assert!(AtomicScope::Device.includes(&AtomicScope::Block));
            assert!(!AtomicScope::Thread.includes(&AtomicScope::Device));
    }

    #[test]
    fn test_cas_validator_success() {
    let cas = CasOperation {
                event_id: 0, address: 0x100, expected: 42, desired: 99,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                scope: AtomicScope::Device,
            };
            let v = CasValidator::new(true);
            assert!(v.validate_cas(&cas, 42, CasOutcome::Success(42)).is_ok());
    }

    #[test]
    fn test_cas_validator_failure() {
    let cas = CasOperation {
                event_id: 0, address: 0x100, expected: 42, desired: 99,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                scope: AtomicScope::Device,
            };
            let v = CasValidator::new(false);
            assert!(v.validate_cas(&cas, 50, CasOutcome::Failure(50)).is_ok());
    }

    #[test]
    fn test_cas_validator_wrong_old() {
    let cas = CasOperation {
                event_id: 0, address: 0x100, expected: 42, desired: 99,
                success_ordering: AtomicOrdering::SeqCst,
                failure_ordering: AtomicOrdering::Relaxed,
                scope: AtomicScope::Device,
            };
            let v = CasValidator::new(true);
            assert!(v.validate_cas(&cas, 42, CasOutcome::Success(99)).is_err());
    }

    #[test]
    fn test_cas_co_constraint() {
    let co = vec![(0, 1), (1, 2)];
            // CAS at event 2, RF from event 0, event 1 intervenes
            assert!(!CasValidator::cas_co_constraint(2, 0, &co, 0x100, &[0, 1, 2]));
            // CAS at event 1, RF from event 0, no intervening
            assert!(CasValidator::cas_co_constraint(1, 0, &co, 0x100, &[0, 1, 2]));
    }

    #[test]
    fn test_fetch_validator() {
    let fetch = FetchOperation {
                event_id: 0, address: 0x100, operand: 5,
                op_kind: FetchOpKind::Add, ordering: AtomicOrdering::SeqCst,
                scope: AtomicScope::Device,
            };
            assert!(FetchValidator::validate_fetch_value(&fetch, 10, 15).is_ok());
            assert!(FetchValidator::validate_fetch_value(&fetch, 10, 16).is_err());
    }

    #[test]
    fn test_fetch_op_kind() {
    assert_eq!(FetchOpKind::Add.apply(10, 5), 15);
            assert_eq!(FetchOpKind::Sub.apply(10, 3), 7);
            assert_eq!(FetchOpKind::And.apply(0xFF, 0x0F), 0x0F);
            assert_eq!(FetchOpKind::Or.apply(0xF0, 0x0F), 0xFF);
            assert_eq!(FetchOpKind::Xor.apply(0xFF, 0xFF), 0);
            assert_eq!(FetchOpKind::Min.apply(10, 5), 5);
            assert_eq!(FetchOpKind::Max.apply(10, 5), 10);
            assert_eq!(FetchOpKind::Exchange.apply(10, 42), 42);
    }

    #[test]
    fn test_atomicity_checker_no_violations() {
    let mut exec = AtomicExecution::new();
            exec.add_event(make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1));
            exec.add_event(make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 1, 2));
            exec.add_rf(0, 1); // conceptual: event 1 reads after event 0
            exec.add_co(0, 1);
            let checker = AtomicityChecker::new(true);
            let violations = checker.check_rmw_atomicity(&exec);
            assert!(violations.is_empty());
    }

    #[test]
    fn test_atomicity_checker_intervening_write() {
    let mut exec = AtomicExecution::new();
            exec.add_event(make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1));
            exec.add_event(make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 0, 1));
            exec.add_event(make_atomic(2, RmwOp::FetchAdd, 0x100, 2, 0, 1, 1, 2));
            // RF: E2 reads from E0, but E1 is CO-between E0 and E2
            exec.add_rf(0, 2);
            exec.add_co(0, 1);
            exec.add_co(1, 2);
            let checker = AtomicityChecker::new(true);
            let violations = checker.check_rmw_atomicity(&exec);
            assert!(!violations.is_empty());
    }

    #[test]
    fn test_gpu_atomic_model_nvidia() {
    let m = GpuAtomicModel::nvidia();
            assert_eq!(m.warp_size, 32);
            assert!(m.supports_system_scope);
    }

    #[test]
    fn test_gpu_atomic_model_amd() {
    let m = GpuAtomicModel::amd();
            assert_eq!(m.warp_size, 64);
            assert!(!m.supports_warp_scope);
    }

    #[test]
    fn test_interference_analysis() {
    let events = vec![
                make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1),
                make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 0, 1),
                make_atomic(2, RmwOp::FetchAdd, 0x200, 2, 0, 1, 0, 1),
            ];
            let ia = InterferenceAnalysis::new(events);
            assert!(ia.has_interference());
            let conflicts = ia.conflict_set();
            assert_eq!(conflicts.len(), 1);
            assert_eq!(conflicts[0], (0, 1));
    }

    #[test]
    fn test_interference_no_conflict() {
    let events = vec![
                make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1),
                make_atomic(1, RmwOp::FetchAdd, 0x200, 1, 0, 1, 0, 1),
            ];
            let ia = InterferenceAnalysis::new(events);
            assert!(!ia.has_interference());
    }

    #[test]
    fn test_rmw_outcome_enumerator() {
    let events = vec![
                make_atomic(0, RmwOp::CompareAndSwap, 0x100, 0, 42, 99, 42, 99),
            ];
            let en = RmwOutcomeEnumerator::new(events);
            let outcomes = en.enumerate_outcomes();
            assert_eq!(outcomes.len(), 2); // success or failure
    }

    #[test]
    fn test_rmw_outcome_enumerator_unconditional() {
    let events = vec![
                make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 5, 10, 15),
            ];
            let en = RmwOutcomeEnumerator::new(events);
            let outcomes = en.enumerate_outcomes();
            assert_eq!(outcomes.len(), 1); // only success
    }

    #[test]
    fn test_rmw_statistics() {
    let mut exec = AtomicExecution::new();
            exec.add_event(make_atomic(0, RmwOp::CompareAndSwap, 0x100, 0, 0, 1, 0, 1));
            exec.add_event(make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 1, 2));
            exec.add_event(make_atomic(2, RmwOp::Exchange, 0x200, 0, 0, 42, 2, 42));
            let stats = rmw_statistics(&exec);
            assert_eq!(stats.total_rmw_ops, 3);
            assert_eq!(stats.cas_ops, 1);
            assert_eq!(stats.fetch_ops, 1);
            assert_eq!(stats.exchange_ops, 1);
            assert_eq!(stats.unique_addresses, 2);
            assert_eq!(stats.unique_threads, 2);
    }

    #[test]
    fn test_atomic_operation_graph() {
    let mut exec = AtomicExecution::new();
            exec.add_event(make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1));
            exec.add_event(make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 1, 2));
            exec.add_co(0, 1);
            let dot = atomic_operation_graph(&exec);
            assert!(dot.contains("digraph"));
            assert!(dot.contains("co"));
    }

    #[test]
    fn test_scope_consistency_ok() {
    let mut exec = AtomicExecution::new();
            let mut e0 = make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1);
            e0.scope = AtomicScope::Device;
            let mut e1 = make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 1, 2);
            e1.scope = AtomicScope::Device;
            exec.add_event(e0);
            exec.add_event(e1);
            exec.add_co(0, 1);
            let checker = AtomicityChecker::new(true);
            let violations = checker.check_scope_consistency(&exec);
            assert!(violations.is_empty());
    }

    #[test]
    fn test_rmw_semantics() {
    let sem = RmwSemantics::new(RmwOp::CompareAndSwap, AtomicOrdering::AcqRel, AtomicScope::Device);
            assert_eq!(sem.read_ordering, AtomicOrdering::Acquire);
            assert_eq!(sem.write_ordering, AtomicOrdering::Release);
    }

    #[test]
    fn test_format_rmw_execution() {
    let mut exec = AtomicExecution::new();
            exec.add_event(make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1));
            let s = format_rmw_execution(&exec);
            assert!(s.contains("RMW Execution"));
    }

    #[test]
    fn test_atomicity_violation_display() {
    let v = AtomicityViolation {
                axiom: AtomicityAxiom::NoIntervening,
                events: vec![0, 1, 2],
                description: "test violation".to_string(),
            };
            let s = format!("{}", v);
            assert!(s.contains("no-intervening"));
            assert!(s.contains("test violation"));
    }

    #[test]
    fn test_warp_level_atomicity() {
    let mut wla = WarpLevelAtomicity::new(0, vec![0, 1, 2, 3]);
            let events = vec![
                make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1),
                make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 1, 2),
                make_atomic(2, RmwOp::FetchAdd, 0x200, 2, 0, 1, 0, 1),
            ];
            wla.coalesce_atomics(&events);
            // Two addresses: 0x100 with events 0,1 and 0x200 with event 2
            assert_eq!(wla.coalesced_atomics.len(), 2);
    }

    #[test]
    fn test_subgroup_atomicity_visibility() {
    let sg = SubgroupAtomicity::new(0, vec![0, 1, 2, 3], AtomicScope::Warp);
            let mut e = make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1);
            e.scope = AtomicScope::Warp;
            let vis = sg.atomic_visibility(&e);
            assert_eq!(vis, vec![0, 1, 2, 3]);
            e.scope = AtomicScope::Thread;
            let vis2 = sg.atomic_visibility(&e);
            assert_eq!(vis2, vec![0]);
    }

    #[test]
    fn test_empty_execution() {
    let exec = AtomicExecution::new();
            let checker = AtomicityChecker::new(true);
            assert!(checker.check_all(&exec).is_empty());
    }

    #[test]
    fn test_conflicts_by_address() {
    let events = vec![
                make_atomic(0, RmwOp::FetchAdd, 0x100, 0, 0, 1, 0, 1),
                make_atomic(1, RmwOp::FetchAdd, 0x100, 1, 0, 1, 0, 1),
                make_atomic(2, RmwOp::FetchAdd, 0x100, 2, 0, 1, 0, 1),
                make_atomic(3, RmwOp::FetchAdd, 0x200, 3, 0, 1, 0, 1),
            ];
            let ia = InterferenceAnalysis::new(events);
            let by_addr = ia.conflicts_by_address();
            assert!(by_addr.contains_key(&0x100));
            assert!(!by_addr.contains_key(&0x200));
    }

    #[test]
    fn test_rmw_op_all() {
    let all = RmwOp::all();
            assert_eq!(all.len(), 10);
    }
    #[test]
    fn test_weakcassemantics_new() {
        let item = WeakCasSemantics::new(0, 0.0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_strongcassemantics_new() {
        let item = StrongCasSemantics::new(false, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_llscmonitor_new() {
        let item = LlScMonitor::new(false, 0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_llscstate_new() {
        let item = LlScState::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomicscheduler_new() {
        let item = AtomicScheduler::new(0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_rmwconflictgraph_new() {
        let item = RmwConflictGraph::new(0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomicorderinglattice_new() {
        let item = AtomicOrderingLattice::new(Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_fenceinsertion_new() {
        let item = FenceInsertion::new(0, "test".to_string(), 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomicreorderanalysis_new() {
        let item = AtomicReorderAnalysis::new(Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_warpatomiccoalescing_new() {
        let item = WarpAtomicCoalescing::new(0, 0, 0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomiccontentiontracker_new() {
        let item = AtomicContentionTracker::new(Vec::new(), 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomicprogressguarantee_new() {
        let item = AtomicProgressGuarantee::new("test".to_string(), 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_atomic_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = atomic_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = atomic_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_atomic_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = atomic_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = atomic_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_atomic_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = atomic_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_atomic_analysis() {
        let mut a = AtomicAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_atomic_iterator() {
        let iter = AtomicResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_atomic_batch_processor() {
        let mut proc = AtomicBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_atomic_histogram() {
        let hist = AtomicHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_atomic_graph() {
        let mut g = AtomicGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_atomic_graph_shortest_path() {
        let mut g = AtomicGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_graph_topo_sort() {
        let mut g = AtomicGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_atomic_graph_components() {
        let mut g = AtomicGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_atomic_cache() {
        let mut cache = AtomicCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_atomic_config() {
        let config = AtomicConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_atomic_report() {
        let mut report = AtomicReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_atomic_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = atomic_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_atomic_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = atomic_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_harmonic_mean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = atomic_harmonic_mean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_atomic_geometric_mean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = atomic_geometric_mean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_atomic_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = atomic_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_atomic_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = atomic_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_atomic_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = atomic_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_atomic_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = atomic_percentile(&data);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_atomic_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = atomic_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_atomic_analysis_normalize() {
        let mut a = AtomicAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_analysis_transpose() {
        let mut a = AtomicAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_analysis_multiply() {
        let mut a = AtomicAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = AtomicAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_analysis_frobenius() {
        let mut a = AtomicAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_analysis_symmetric() {
        let mut a = AtomicAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_atomic_graph_dot() {
        let mut g = AtomicGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_atomic_histogram_render() {
        let hist = AtomicHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_atomic_batch_reset() {
        let mut proc = AtomicBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_atomic_graph_remove_edge() {
        let mut g = AtomicGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_atomicity_dense_matrix_new() {
        let m = AtomicityDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_atomicity_dense_matrix_identity() {
        let m = AtomicityDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_mul() {
        let a = AtomicityDenseMatrix::identity(2);
        let b = AtomicityDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_transpose() {
        let a = AtomicityDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_det_2x2() {
        let m = AtomicityDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_det_3x3() {
        let m = AtomicityDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_inverse_2x2() {
        let m = AtomicityDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_power() {
        let m = AtomicityDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_rank() {
        let m = AtomicityDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_atomicity_dense_matrix_solve() {
        let a = AtomicityDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_atomicity_dense_matrix_lu() {
        let a = AtomicityDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_eigenvalues() {
        let m = AtomicityDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_kronecker() {
        let a = AtomicityDenseMatrix::identity(2);
        let b = AtomicityDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_atomicity_dense_matrix_hadamard() {
        let a = AtomicityDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = AtomicityDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_interval() {
        let a = AtomicityInterval::new(1.0, 3.0);
        let b = AtomicityInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_interval_mul() {
        let a = AtomicityInterval::new(-2.0, 3.0);
        let b = AtomicityInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_interval_hull() {
        let a = AtomicityInterval::new(1.0, 3.0);
        let b = AtomicityInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_state_machine() {
        let mut sm = AtomicityStateMachine::new();
        assert_eq!(*sm.state(), AtomicityState::Ready);
        assert!(sm.transition(AtomicityState::Acquiring));
        assert_eq!(*sm.state(), AtomicityState::Acquiring);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_atomicity_state_machine_invalid() {
        let mut sm = AtomicityStateMachine::new();
        let last_state = AtomicityState::Failed;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_atomicity_state_machine_reset() {
        let mut sm = AtomicityStateMachine::new();
        sm.transition(AtomicityState::Acquiring);
        sm.reset();
        assert_eq!(*sm.state(), AtomicityState::Ready);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_atomicity_ring_buffer() {
        let mut rb = AtomicityRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_ring_buffer_to_vec() {
        let mut rb = AtomicityRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_disjoint_set() {
        let mut ds = AtomicityDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_atomicity_disjoint_set_components() {
        let mut ds = AtomicityDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_atomicity_sorted_list() {
        let mut sl = AtomicitySortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sorted_list_remove() {
        let mut sl = AtomicitySortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_atomicity_ema() {
        let mut ema = AtomicityEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_bloom_filter() {
        let mut bf = AtomicityBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_atomicity_trie() {
        let mut trie = AtomicityTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_atomicity_dense_matrix_sym() {
        let m = AtomicityDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_atomicity_dense_matrix_diag() {
        let m = AtomicityDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_atomicity_dense_matrix_upper_tri() {
        let m = AtomicityDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_atomicity_dense_matrix_outer() {
        let m = AtomicityDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dense_matrix_submatrix() {
        let m = AtomicityDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_priority_queue() {
        let mut pq = AtomicityPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_atomicity_accumulator() {
        let mut acc = AtomicityAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_accumulator_merge() {
        let mut a = AtomicityAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = AtomicityAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sparse_matrix() {
        let mut m = AtomicitySparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sparse_mul_vec() {
        let mut m = AtomicitySparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sparse_transpose() {
        let mut m = AtomicitySparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_eval() {
        let p = AtomicityPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_add() {
        let a = AtomicityPolynomial::new(vec![1.0, 2.0]);
        let b = AtomicityPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_mul() {
        let a = AtomicityPolynomial::new(vec![1.0, 1.0]);
        let b = AtomicityPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_deriv() {
        let p = AtomicityPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_integral() {
        let p = AtomicityPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_roots() {
        let p = AtomicityPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_atomicity_polynomial_newton() {
        let p = AtomicityPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_atomicity_polynomial_compose() {
        let p = AtomicityPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = AtomicityPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_rng() {
        let mut rng = AtomicityRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_atomicity_rng_gaussian() {
        let mut rng = AtomicityRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_atomicity_timer() {
        let mut timer = AtomicityTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_bitvector() {
        let mut bv = AtomicityBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_atomicity_bitvector_ops() {
        let mut a = AtomicityBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = AtomicityBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_atomicity_bitvector_jaccard() {
        let mut a = AtomicityBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = AtomicityBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_priority_queue_empty() {
        let mut pq = AtomicityPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_atomicity_sparse_add() {
        let mut a = AtomicitySparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = AtomicitySparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_rng_shuffle() {
        let mut rng = AtomicityRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_polynomial_display() {
        let p = AtomicityPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_atomicity_polynomial_monomial() {
        let m = AtomicityPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_timer_percentiles() {
        let mut timer = AtomicityTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_atomicity_accumulator_cv() {
        let mut acc = AtomicityAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sparse_diagonal() {
        let mut m = AtomicitySparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_lru_cache() {
        let mut cache = AtomicityLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_atomicity_lru_hit_rate() {
        let mut cache = AtomicityLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_graph_coloring() {
        let mut gc = AtomicityGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_atomicity_graph_coloring_complete() {
        let mut gc = AtomicityGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_atomicity_topk() {
        let mut tk = AtomicityTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sliding_window() {
        let mut sw = AtomicitySlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_atomicity_sliding_window_trend() {
        let mut sw = AtomicitySlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_atomicity_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = AtomicityConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_atomicity_confusion_f1() {
        let cm = AtomicityConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_atomicity_cosine_similarity() {
        let s = atomicity_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = atomicity_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_euclidean_distance() {
        let d = atomicity_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sigmoid() {
        let s = atomicity_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = atomicity_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_atomicity_softmax() {
        let sm = atomicity_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_atomicity_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = atomicity_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_normalize() {
        let v = atomicity_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_lerp() {
        assert!((atomicity_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((atomicity_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((atomicity_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_clamp() {
        assert!((atomicity_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((atomicity_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((atomicity_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_cross_product() {
        let c = atomicity_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dot_product() {
        let d = atomicity_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = atomicity_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = atomicity_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_logsumexp() {
        let lse = atomicity_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_atomicity_feature_scaler() {
        let mut scaler = AtomicityFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_feature_scaler_inverse() {
        let mut scaler = AtomicityFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_linear_regression() {
        let mut lr = AtomicityLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_linear_regression_predict() {
        let mut lr = AtomicityLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_weighted_graph() {
        let mut g = AtomicityWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_weighted_graph_mst() {
        let mut g = AtomicityWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = atomicity_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_cumsum() {
        let cs = atomicity_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_diff() {
        let d = atomicity_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_autocorrelation() {
        let ac = atomicity_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_dft_magnitude() {
        let mags = atomicity_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_atomicity_integrate_trapezoid() {
        let area = atomicity_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_convolve() {
        let c = atomicity_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_weighted_graph_clustering() {
        let mut g = AtomicityWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_histogram() {
        let h = AtomicityHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 5);
        assert_eq!(h.total, 5);
        assert_eq!(h.num_bins(), 5);
    }

    #[test]
    fn test_atomicity_histogram_cumulative() {
        let h = AtomicityHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_atomicity_histogram_entropy() {
        let h = AtomicityHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_atomicity_aabb() {
        let bb = AtomicityAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_aabb_intersects() {
        let a = AtomicityAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = AtomicityAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = AtomicityAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_atomicity_quadtree() {
        let bb = AtomicityAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = AtomicityQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(AtomicityPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_atomicity_quadtree_query() {
        let bb = AtomicityAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = AtomicityQuadTree::new(bb, 2, 8);
        qt.insert(AtomicityPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(AtomicityPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = AtomicityAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_atomicity_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = atomicity_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = atomicity_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = atomicity_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((atomicity_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_identity() {
        let id = atomicity_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = atomicity_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_atomicity_running_stats() {
        let mut s = AtomicityRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_running_stats_merge() {
        let mut a = AtomicityRunningStats::new();
        let mut b = AtomicityRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_running_stats_cv() {
        let mut s = AtomicityRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_atomicity_percentile_at() {
        let p50 = atomicity_percentile_at(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0);
        assert!((p50 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_iqr() {
        let iqr = atomicity_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_atomicity_outliers() {
        let outliers = atomicity_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_atomicity_zscore() {
        let z = atomicity_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_atomicity_rank() {
        let r = atomicity_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_spearman() {
        let rho = atomicity_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_geomean() {
        let gm = atomicity_geomean(&[1.0, 2.0, 4.0, 8.0]);
        assert!((gm - (1.0 * 2.0 * 4.0 * 8.0_f64).powf(0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_atomicity_harmmean() {
        let hm = atomicity_harmmean(&[1.0, 2.0, 4.0]);
        let expected = 3.0 / (1.0 + 0.5 + 0.25);
        assert!((hm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_sample_skewness_symmetric() {
        let s = atomicity_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_atomicity_excess_kurtosis() {
        let k = atomicity_excess_kurtosis(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(k.is_finite());
    }

    #[test]
    fn test_atomicity_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = atomicity_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_atomicity_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = atomicity_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }


}