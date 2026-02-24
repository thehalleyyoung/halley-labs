/// Write serialization: total order on writes, coherence checking,
/// and serialization verification for LITMUS∞ memory model verification.
///
/// Implements per-location write serialization, from-read computation,
/// coherence checking (SC-per-location), total order enumeration,
/// write coalescing analysis, and scope-aware serialization.
#[allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use super::execution::{EventId, ThreadId, Address, Value, OpType, Scope, Event, BitMatrix, ExecutionGraph, Relation};

// ═══════════════════════════════════════════════════════════════════════════
// WriteEvent and WriteSet
// ═══════════════════════════════════════════════════════════════════════════

/// A write event enriched with serialization metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WriteEvent {
    /// The original event index.
    pub event_id: EventId,
    /// Thread that performed the write.
    pub thread: ThreadId,
    /// Address written to.
    pub address: Address,
    /// Value written.
    pub value: Value,
    /// Position in the serialization order (None if not yet serialized).
    pub serial_index: Option<usize>,
    /// Whether this is an initialization write.
    pub is_init: bool,
    /// Whether this is the final write (last in the serial order).
    pub is_final: bool,
    /// Scope of the write operation.
    pub scope: Scope,
}

impl WriteEvent {
    /// Create a new write event from an Event.
    pub fn from_event(event: &Event, event_id: EventId) -> Option<Self> {
        if event.is_write() {
            Some(Self {
                event_id,
                thread: event.thread,
                address: event.address,
                value: event.value,
                serial_index: None,
                is_init: false,
                is_final: false,
                scope: event.scope,
            })
        } else {
            None
        }
    }

    /// Mark as initialization write.
    pub fn mark_init(mut self) -> Self {
        self.is_init = true;
        self
    }

    /// Mark as final write.
    pub fn mark_final(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Set the serial index.
    pub fn set_serial_index(&mut self, idx: usize) {
        self.serial_index = Some(idx);
    }
}

impl fmt::Display for WriteEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "W(E{}, T{}, @{:#x}, v={}", self.event_id, self.thread, self.address, self.value)?;
        if self.is_init { write!(f, ", init")?; }
        if self.is_final { write!(f, ", final")?; }
        if let Some(idx) = self.serial_index { write!(f, ", #{}", idx)?; }
        write!(f, ")")
    }
}

/// A collection of write events grouped by address.
#[derive(Debug, Clone)]
pub struct WriteSet {
    /// All write events.
    pub writes: Vec<WriteEvent>,
    /// Writes indexed by address.
    by_address: HashMap<Address, Vec<usize>>,
    /// Writes indexed by thread.
    by_thread: HashMap<ThreadId, Vec<usize>>,
}

impl WriteSet {
    /// Create a new empty write set.
    pub fn new() -> Self {
        Self {
            writes: Vec::new(),
            by_address: HashMap::new(),
            by_thread: HashMap::new(),
        }
    }

    /// Create from an execution graph, extracting all writes.
    pub fn from_execution(exec: &ExecutionGraph) -> Self {
        let mut ws = Self::new();
        for (idx, event) in exec.events.iter().enumerate() {
            if let Some(we) = WriteEvent::from_event(event, idx) {
                ws.add_write(we);
            }
        }
        ws
    }

    /// Add a write event.
    pub fn add_write(&mut self, we: WriteEvent) {
        let idx = self.writes.len();
        self.by_address.entry(we.address).or_default().push(idx);
        self.by_thread.entry(we.thread).or_default().push(idx);
        self.writes.push(we);
    }

    /// Get writes to a specific address.
    pub fn writes_to_address(&self, addr: Address) -> Vec<&WriteEvent> {
        self.by_address.get(&addr)
            .map(|indices| indices.iter().map(|&i| &self.writes[i]).collect())
            .unwrap_or_default()
    }

    /// Get all writes.
    pub fn all_writes(&self) -> &[WriteEvent] {
        &self.writes
    }

    /// Get writes by a specific thread.
    pub fn filter_by_thread(&self, tid: ThreadId) -> Vec<&WriteEvent> {
        self.by_thread.get(&tid)
            .map(|indices| indices.iter().map(|&i| &self.writes[i]).collect())
            .unwrap_or_default()
    }

    /// Get all unique addresses written to.
    pub fn addresses(&self) -> Vec<Address> {
        self.by_address.keys().copied().collect()
    }

    /// Group writes by address.
    pub fn group_by_address(&self) -> HashMap<Address, Vec<&WriteEvent>> {
        let mut groups = HashMap::new();
        for (addr, indices) in &self.by_address {
            groups.insert(*addr, indices.iter().map(|&i| &self.writes[i]).collect());
        }
        groups
    }

    /// Number of writes.
    pub fn len(&self) -> usize { self.writes.len() }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool { self.writes.is_empty() }
}

impl Default for WriteSet {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════
// WriteSerialization
// ═══════════════════════════════════════════════════════════════════════════

/// A total order on writes to a single address.
#[derive(Debug, Clone)]
pub struct SerialOrder {
    /// Address this order applies to.
    pub address: Address,
    /// Event IDs in serial order (earliest first).
    pub order: Vec<EventId>,
    /// Map from event ID to position in the order.
    position_map: HashMap<EventId, usize>,
}

impl SerialOrder {
    /// Create a new empty serial order for an address.
    pub fn new(address: Address) -> Self {
        Self {
            address,
            order: Vec::new(),
            position_map: HashMap::new(),
        }
    }

    /// Create from an ordered list of event IDs.
    pub fn from_order(address: Address, order: Vec<EventId>) -> Self {
        let position_map: HashMap<EventId, usize> = order.iter()
            .enumerate()
            .map(|(pos, &eid)| (eid, pos))
            .collect();
        Self { address, order, position_map }
    }

    /// Add a write to the end of the order.
    pub fn append(&mut self, event_id: EventId) {
        let pos = self.order.len();
        self.position_map.insert(event_id, pos);
        self.order.push(event_id);
    }

    /// Insert a write at a specific position.
    pub fn insert_at(&mut self, pos: usize, event_id: EventId) {
        self.order.insert(pos, event_id);
        self.rebuild_position_map();
    }

    fn rebuild_position_map(&mut self) {
        self.position_map.clear();
        for (pos, &eid) in self.order.iter().enumerate() {
            self.position_map.insert(eid, pos);
        }
    }

    /// Get the position of an event in the order.
    pub fn position(&self, event_id: EventId) -> Option<usize> {
        self.position_map.get(&event_id).copied()
    }

    /// Check whether event a is before event b in the order.
    pub fn is_before(&self, a: EventId, b: EventId) -> bool {
        match (self.position(a), self.position(b)) {
            (Some(pa), Some(pb)) => pa < pb,
            _ => false,
        }
    }

    /// Get the immediate predecessor of an event.
    pub fn immediate_predecessor(&self, event_id: EventId) -> Option<EventId> {
        self.position(event_id)
            .filter(|&p| p > 0)
            .map(|p| self.order[p - 1])
    }

    /// Get the immediate successor of an event.
    pub fn immediate_successor(&self, event_id: EventId) -> Option<EventId> {
        self.position(event_id)
            .filter(|&p| p + 1 < self.order.len())
            .map(|p| self.order[p + 1])
    }

    /// Validate the order: no duplicates, all writes present.
    pub fn validate(&self, expected_writes: &[EventId]) -> bool {
        if self.order.len() != expected_writes.len() { return false; }
        let order_set: HashSet<EventId> = self.order.iter().copied().collect();
        let expected_set: HashSet<EventId> = expected_writes.iter().copied().collect();
        order_set == expected_set
    }

    /// Convert to a BitMatrix relation (n is total number of events).
    pub fn to_bit_matrix(&self, n: usize) -> BitMatrix {
        let mut mat = BitMatrix::new(n);
        for i in 0..self.order.len() {
            for j in (i + 1)..self.order.len() {
                mat.set(self.order[i], self.order[j], true);
            }
        }
        mat
    }

    /// Number of writes in the order.
    pub fn len(&self) -> usize { self.order.len() }

    /// Whether the order is empty.
    pub fn is_empty(&self) -> bool { self.order.is_empty() }
}

impl fmt::Display for SerialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CO(@{:#x}): ", self.address)?;
        for (i, &eid) in self.order.iter().enumerate() {
            if i > 0 { write!(f, " < ")?; }
            write!(f, "E{}", eid)?;
        }
        Ok(())
    }
}

/// Write serialization: managing total orders on writes per address.
#[derive(Debug, Clone)]
pub struct WriteSerialization {
    /// Per-address serial orders.
    orders: HashMap<Address, SerialOrder>,
    /// Total number of events.
    num_events: usize,
    /// Combined coherence order as a BitMatrix.
    co_matrix: BitMatrix,
    /// Whether the CO matrix has been computed.
    co_computed: bool,
}

impl WriteSerialization {
    /// Create a new empty write serialization.
    pub fn new(num_events: usize) -> Self {
        Self {
            orders: HashMap::new(),
            num_events,
            co_matrix: BitMatrix::new(num_events),
            co_computed: false,
        }
    }

    /// Build from a coherence order BitMatrix and events.
    pub fn from_coherence_order(events: &[Event], co: &BitMatrix) -> Self {
        let n = events.len();
        let mut ws = Self::new(n);
        // Group writes by address
        let mut writes_per_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        for (idx, ev) in events.iter().enumerate() {
            if ev.is_write() {
                writes_per_addr.entry(ev.address).or_default().push(idx);
            }
        }
        // For each address, sort by CO
        for (addr, mut writes) in writes_per_addr {
            writes.sort_by(|&a, &b| {
                if co.get(a, b) { std::cmp::Ordering::Less }
                else if co.get(b, a) { std::cmp::Ordering::Greater }
                else { a.cmp(&b) }
            });
            ws.orders.insert(addr, SerialOrder::from_order(addr, writes));
        }
        ws.co_matrix = co.clone();
        ws.co_computed = true;
        ws
    }

    /// Set the order for an address.
    pub fn set_order(&mut self, addr: Address, order: Vec<EventId>) {
        self.orders.insert(addr, SerialOrder::from_order(addr, order));
        self.co_computed = false;
    }

    /// Get the order for an address.
    pub fn get_order(&self, addr: Address) -> Option<&SerialOrder> {
        self.orders.get(&addr)
    }

    /// Check whether event a is CO-before event b.
    pub fn is_before(&self, a: EventId, b: EventId, addr: Address) -> bool {
        self.orders.get(&addr).map_or(false, |o| o.is_before(a, b))
    }

    /// Get the combined CO matrix.
    pub fn get_co_matrix(&mut self) -> &BitMatrix {
        if !self.co_computed {
            self.compute_co_matrix();
        }
        &self.co_matrix
    }

    fn compute_co_matrix(&mut self) {
        self.co_matrix = BitMatrix::new(self.num_events);
        for order in self.orders.values() {
            let mat = order.to_bit_matrix(self.num_events);
            self.co_matrix = self.co_matrix.union(&mat);
        }
        self.co_computed = true;
    }

    /// Get all addresses with serialization orders.
    pub fn addresses(&self) -> Vec<Address> {
        self.orders.keys().copied().collect()
    }

    /// Validate all orders.
    pub fn validate_all(&self, write_set: &WriteSet) -> bool {
        for addr in self.orders.keys() {
            let writes: Vec<EventId> = write_set.writes_to_address(*addr)
                .iter().map(|w| w.event_id).collect();
            if let Some(order) = self.orders.get(addr) {
                if !order.validate(&writes) { return false; }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CoherenceChecker
// ═══════════════════════════════════════════════════════════════════════════

/// Type of coherence violation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoherenceViolationType {
    /// CO-RF inconsistency: a read sees a write that is CO-after another write.
    CoRfInconsistency,
    /// CO-FR inconsistency: from-read violates coherence order.
    CoFrInconsistency,
    /// CO-PO inconsistency: program order contradicts coherence order.
    CoPoInconsistency,
    /// CO not total: two writes to the same address are not CO-ordered.
    CoNotTotal,
    /// CO cycle detected.
    CoCycle,
}

impl fmt::Display for CoherenceViolationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoherenceViolationType::CoRfInconsistency => write!(f, "CO-RF"),
            CoherenceViolationType::CoFrInconsistency => write!(f, "CO-FR"),
            CoherenceViolationType::CoPoInconsistency => write!(f, "CO-PO"),
            CoherenceViolationType::CoNotTotal => write!(f, "CO-NotTotal"),
            CoherenceViolationType::CoCycle => write!(f, "CO-Cycle"),
        }
    }
}

/// A coherence violation.
#[derive(Debug, Clone)]
pub struct CoherenceViolation {
    /// Type of violation.
    pub violation_type: CoherenceViolationType,
    /// Description of the violation.
    pub description: String,
    /// Events involved.
    pub events: Vec<EventId>,
    /// Address involved.
    pub address: Address,
}

impl fmt::Display for CoherenceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} @{:#x}: {} (events: {:?})",
            self.violation_type, self.address, self.description, self.events)
    }
}

/// Checker for coherence (SC-per-location) in executions.
#[derive(Debug, Clone)]
pub struct CoherenceChecker {
    /// Events in the execution.
    events: Vec<Event>,
    /// Program-order relation.
    po: BitMatrix,
    /// Reads-from relation.
    rf: BitMatrix,
    /// Coherence order.
    co: BitMatrix,
    /// From-read relation.
    fr: BitMatrix,
    /// Detected violations.
    violations: Vec<CoherenceViolation>,
}

impl CoherenceChecker {
    /// Create a new coherence checker.
    pub fn new(events: Vec<Event>, po: BitMatrix, rf: BitMatrix, co: BitMatrix) -> Self {
        let n = events.len();
        let rf_inv = rf.inverse();
        let fr = rf_inv.compose(&co);
        Self { events, po, rf, co, fr, violations: Vec::new() }
    }

    /// Run all coherence checks.
    pub fn check_coherence(&mut self) -> &[CoherenceViolation] {
        self.violations.clear();
        self.check_co_totality();
        self.check_co_acyclicity();
        self.check_co_rf_consistency();
        self.check_co_fr_consistency();
        self.check_co_po_consistency();
        &self.violations
    }

    /// Check that CO is total per address.
    pub fn check_co_totality(&mut self) {
        let n = self.events.len();
        let mut writes_per_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.is_write() {
                writes_per_addr.entry(ev.address).or_default().push(idx);
            }
        }
        for (addr, writes) in &writes_per_addr {
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    let a = writes[i];
                    let b = writes[j];
                    if !self.co.get(a, b) && !self.co.get(b, a) {
                        self.violations.push(CoherenceViolation {
                            violation_type: CoherenceViolationType::CoNotTotal,
                            description: format!("W{} and W{} not CO-ordered", a, b),
                            events: vec![a, b],
                            address: *addr,
                        });
                    }
                }
            }
        }
    }

    /// Check that CO has no cycles.
    pub fn check_co_acyclicity(&mut self) {
        let n = self.events.len();
        let closure = self.co.transitive_closure();
        for i in 0..n {
            if closure.get(i, i) && self.events[i].is_write() {
                self.violations.push(CoherenceViolation {
                    violation_type: CoherenceViolationType::CoCycle,
                    description: format!("CO cycle at E{}", i),
                    events: vec![i],
                    address: self.events[i].address,
                });
            }
        }
    }

    /// Check CO-RF consistency: if rf(w, r) and co(w', w), then not rf(w', r).
    pub fn check_co_rf_consistency(&mut self) {
        let n = self.events.len();
        for w in 0..n {
            for r in 0..n {
                if self.rf.get(w, r) {
                    // For each w' such that co(w, w')
                    for w_prime in 0..n {
                        if self.co.get(w, w_prime) && self.rf.get(w_prime, r) {
                            self.violations.push(CoherenceViolation {
                                violation_type: CoherenceViolationType::CoRfInconsistency,
                                description: format!(
                                    "rf(E{},E{}) but co(E{},E{}) and rf(E{},E{})",
                                    w, r, w, w_prime, w_prime, r
                                ),
                                events: vec![w, r, w_prime],
                                address: self.events[w].address,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Check CO-FR consistency: no cycles in co ∪ fr.
    pub fn check_co_fr_consistency(&mut self) {
        let co_fr = self.co.union(&self.fr);
        let closure = co_fr.transitive_closure();
        let n = self.events.len();
        for i in 0..n {
            if closure.get(i, i) {
                self.violations.push(CoherenceViolation {
                    violation_type: CoherenceViolationType::CoFrInconsistency,
                    description: format!("CO-FR cycle at E{}", i),
                    events: vec![i],
                    address: self.events[i].address,
                });
            }
        }
    }

    /// Check CO-PO consistency for SC-per-location.
    pub fn check_co_po_consistency(&mut self) {
        let n = self.events.len();
        for i in 0..n {
            for j in 0..n {
                if self.po.get(i, j)
                    && self.events[i].is_write()
                    && self.events[j].is_write()
                    && self.events[i].address == self.events[j].address
                    && self.co.get(j, i)
                {
                    self.violations.push(CoherenceViolation {
                        violation_type: CoherenceViolationType::CoPoInconsistency,
                        description: format!("po(E{},E{}) but co(E{},E{})", i, j, j, i),
                        events: vec![i, j],
                        address: self.events[i].address,
                    });
                }
            }
        }
    }

    /// Find all coherence violations.
    pub fn find_coherence_violations(&self) -> &[CoherenceViolation] {
        &self.violations
    }

    /// Check if execution is coherent.
    pub fn is_coherent(&self) -> bool { self.violations.is_empty() }

    /// Enumerate valid CO orders consistent with constraints.
    pub fn enumerate_valid_orders(&self, addr: Address) -> Vec<Vec<EventId>> {
        let writes: Vec<EventId> = self.events.iter().enumerate()
            .filter(|(_, e)| e.is_write() && e.address == addr)
            .map(|(i, _)| i)
            .collect();
        let mut results = Vec::new();
        let mut perm = writes.clone();
        self.permute_and_check(&mut perm, 0, &writes, &mut results);
        results
    }

    fn permute_and_check(
        &self,
        perm: &mut Vec<EventId>,
        start: usize,
        _writes: &[EventId],
        results: &mut Vec<Vec<EventId>>,
    ) {
        if start == perm.len() {
            if self.is_valid_order(perm) {
                results.push(perm.clone());
            }
            return;
        }
        for i in start..perm.len() {
            perm.swap(start, i);
            self.permute_and_check(perm, start + 1, _writes, results);
            perm.swap(start, i);
        }
    }

    fn is_valid_order(&self, order: &[EventId]) -> bool {
        // Check PO consistency
        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                if self.po.get(order[j], order[i]) {
                    return false;
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TotalOrderEnumerator
// ═══════════════════════════════════════════════════════════════════════════

/// Constraint on total order enumeration.
#[derive(Debug, Clone)]
pub enum OrderConstraint {
    /// a must come before b.
    Before(EventId, EventId),
    /// a must be immediately before b.
    ImmediatelyBefore(EventId, EventId),
    /// a must not come immediately before b.
    NotImmediatelyBefore(EventId, EventId),
}

/// Enumerator for valid total orders on writes.
#[derive(Debug, Clone)]
pub struct TotalOrderEnumerator {
    /// Write event IDs to order.
    writes: Vec<EventId>,
    /// Partial order constraints (must-before).
    constraints: Vec<(EventId, EventId)>,
    /// RF constraints: rf(w, r) means w must be the last write before r sees it.
    rf_constraints: Vec<(EventId, EventId)>,
    /// FR constraints derived from RF and CO.
    fr_constraints: Vec<(EventId, EventId)>,
}

impl TotalOrderEnumerator {
    /// Create a new enumerator.
    pub fn new(writes: Vec<EventId>) -> Self {
        Self {
            writes,
            constraints: Vec::new(),
            rf_constraints: Vec::new(),
            fr_constraints: Vec::new(),
        }
    }

    /// Add a partial order constraint: a must come before b.
    pub fn add_constraint(&mut self, before: EventId, after: EventId) {
        self.constraints.push((before, after));
    }

    /// Add an RF constraint.
    pub fn add_rf_constraint(&mut self, write: EventId, read: EventId) {
        self.rf_constraints.push((write, read));
    }

    /// Add an FR constraint.
    pub fn add_fr_constraint(&mut self, from: EventId, to: EventId) {
        self.fr_constraints.push((from, to));
    }

    /// Enumerate all valid total orders.
    pub fn enumerate(&self) -> Vec<Vec<EventId>> {
        let mut results = Vec::new();
        let mut current = self.writes.clone();
        self.enumerate_recursive(&mut current, 0, &mut results);
        results
    }

    fn enumerate_recursive(
        &self,
        current: &mut Vec<EventId>,
        start: usize,
        results: &mut Vec<Vec<EventId>>,
    ) {
        if start == current.len() {
            if self.is_consistent(current) {
                results.push(current.clone());
            }
            return;
        }
        for i in start..current.len() {
            current.swap(start, i);
            if self.is_prefix_consistent(current, start) {
                self.enumerate_recursive(current, start + 1, results);
            }
            current.swap(start, i);
        }
    }

    /// Check if a complete order is consistent with all constraints.
    pub fn is_consistent(&self, order: &[EventId]) -> bool {
        let pos: HashMap<EventId, usize> = order.iter()
            .enumerate().map(|(i, &e)| (e, i)).collect();
        for &(before, after) in &self.constraints {
            match (pos.get(&before), pos.get(&after)) {
                (Some(&pb), Some(&pa)) if pb >= pa => return false,
                _ => {}
            }
        }
        true
    }

    /// Check prefix consistency for pruning.
    fn is_prefix_consistent(&self, prefix: &[EventId], up_to: usize) -> bool {
        let pos: HashMap<EventId, usize> = prefix[..=up_to].iter()
            .enumerate().map(|(i, &e)| (e, i)).collect();
        for &(before, after) in &self.constraints {
            if let (Some(&pb), Some(&pa)) = (pos.get(&before), pos.get(&after)) {
                if pb > pa { return false; }
            }
            // If after is placed but before is not yet placed, violation
            if pos.contains_key(&after) && !pos.contains_key(&before) {
                return false;
            }
        }
        true
    }

    /// Count the number of valid total orders.
    pub fn count_valid_orders(&self) -> usize {
        self.enumerate().len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SerializationVerifier
// ═══════════════════════════════════════════════════════════════════════════

/// Result of serialization verification.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether the serialization is valid.
    pub is_valid: bool,
    /// List of failures.
    pub failures: Vec<String>,
    /// Per-address results.
    pub per_address: HashMap<Address, bool>,
    /// Summary statistics.
    pub total_writes: usize,
    /// Number of addresses checked.
    pub addresses_checked: usize,
}

impl VerificationReport {
    fn new() -> Self {
        Self {
            is_valid: true,
            failures: Vec::new(),
            per_address: HashMap::new(),
            total_writes: 0,
            addresses_checked: 0,
        }
    }

    fn add_failure(&mut self, msg: String) {
        self.is_valid = false;
        self.failures.push(msg);
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Serialization Verification: {}", if self.is_valid { "PASS" } else { "FAIL" })?;
        writeln!(f, "  Writes: {}, Addresses: {}", self.total_writes, self.addresses_checked)?;
        for failure in &self.failures {
            writeln!(f, "  FAIL: {}", failure)?;
        }
        Ok(())
    }
}

/// Verifier for write serialization axioms.
#[derive(Debug, Clone)]
pub struct SerializationVerifier {
    /// Events.
    events: Vec<Event>,
    /// Write serialization.
    serialization: WriteSerialization,
    /// Reads-from relation.
    rf: BitMatrix,
    /// Program order.
    po: BitMatrix,
}

impl SerializationVerifier {
    /// Create a new verifier.
    pub fn new(
        events: Vec<Event>,
        serialization: WriteSerialization,
        rf: BitMatrix,
        po: BitMatrix,
    ) -> Self {
        Self { events, serialization, rf, po }
    }

    /// Run full verification.
    pub fn verify(&self) -> VerificationReport {
        let mut report = VerificationReport::new();
        for addr in self.serialization.addresses() {
            report.addresses_checked += 1;
            let result = self.verify_per_address(addr, &mut report);
            report.per_address.insert(addr, result);
        }
        self.check_co_totality(&mut report);
        self.check_co_transitivity(&mut report);
        self.check_co_irreflexivity(&mut report);
        self.check_atomicity(&mut report);
        self.check_no_future_read(&mut report);
        report
    }

    /// Verify serialization for a single address.
    pub fn verify_per_address(&self, addr: Address, report: &mut VerificationReport) -> bool {
        let order = match self.serialization.get_order(addr) {
            Some(o) => o,
            None => return true,
        };
        report.total_writes += order.len();
        let mut valid = true;
        // Check each read sees the correct write
        let n = self.events.len();
        for r in 0..n {
            if !self.events[r].is_read() || self.events[r].address != addr {
                continue;
            }
            for w in 0..n {
                if self.rf.get(w, r) {
                    // w must be the latest write before r in some linearization
                    // Check: no w' such that co(w, w') and w' is before r
                    for w_prime in 0..n {
                        if w_prime != w && self.events[w_prime].is_write()
                            && self.events[w_prime].address == addr
                            && order.is_before(w, w_prime)
                            && self.po.get(w_prime, r)
                        {
                            report.add_failure(format!(
                                "Read E{} sees E{} but E{} is co-after and po-before @{:#x}",
                                r, w, w_prime, addr
                            ));
                            valid = false;
                        }
                    }
                }
            }
        }
        valid
    }

    /// Check CO totality: for each address, all writes are CO-ordered.
    pub fn check_co_totality(&self, report: &mut VerificationReport) {
        for addr in self.serialization.addresses() {
            let writes: Vec<EventId> = self.events.iter().enumerate()
                .filter(|(_, e)| e.is_write() && e.address == addr)
                .map(|(i, _)| i)
                .collect();
            if let Some(order) = self.serialization.get_order(addr) {
                for &w in &writes {
                    if order.position(w).is_none() {
                        report.add_failure(format!("Write E{} not in CO for @{:#x}", w, addr));
                    }
                }
            }
        }
    }

    /// Check CO transitivity.
    pub fn check_co_transitivity(&self, report: &mut VerificationReport) {
        for addr in self.serialization.addresses() {
            if let Some(order) = self.serialization.get_order(addr) {
                for i in 0..order.len() {
                    for j in (i + 1)..order.len() {
                        for k in (j + 1)..order.len() {
                            let a = order.order[i];
                            let b = order.order[j];
                            let c = order.order[k];
                            if !order.is_before(a, c) {
                                report.add_failure(format!(
                                    "CO transitivity: E{}<E{}<E{} but not E{}<E{} @{:#x}",
                                    a, b, c, a, c, addr
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check CO irreflexivity.
    pub fn check_co_irreflexivity(&self, report: &mut VerificationReport) {
        for addr in self.serialization.addresses() {
            if let Some(order) = self.serialization.get_order(addr) {
                for &w in &order.order {
                    if order.is_before(w, w) {
                        report.add_failure(format!("CO reflexive: E{} < E{} @{:#x}", w, w, addr));
                    }
                }
            }
        }
    }

    /// Check atomicity of RMW operations.
    pub fn check_atomicity(&self, report: &mut VerificationReport) {
        let n = self.events.len();
        for i in 0..n {
            if self.events[i].is_rmw() {
                // The read part must see the immediately preceding write in CO
                // Simplified: check that no other write intervenes
                for w in 0..n {
                    if self.rf.get(w, i) {
                        let addr = self.events[i].address;
                        if let Some(order) = self.serialization.get_order(addr) {
                            if let Some(succ) = order.immediate_successor(w) {
                                if succ != i {
                                    report.add_failure(format!(
                                        "RMW atomicity: E{} reads E{} but E{} intervenes @{:#x}",
                                        i, w, succ, addr
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check no-future-read: a read cannot see a write that is co-after the latest po-preceding write.
    pub fn check_no_future_read(&self, report: &mut VerificationReport) {
        let n = self.events.len();
        for r in 0..n {
            if !self.events[r].is_read() { continue; }
            let addr = self.events[r].address;
            for w in 0..n {
                if !self.rf.get(w, r) { continue; }
                // Find the latest write to addr that is po-before r
                let mut latest_po_write: Option<EventId> = None;
                for w2 in 0..n {
                    if self.events[w2].is_write()
                        && self.events[w2].address == addr
                        && self.po.get(w2, r)
                    {
                        latest_po_write = Some(w2);
                    }
                }
                if let Some(lpw) = latest_po_write {
                    if let Some(order) = self.serialization.get_order(addr) {
                        if order.is_before(w, lpw) {
                            report.add_failure(format!(
                                "Future read: E{} reads E{} but E{} is po-before and co-after @{:#x}",
                                r, w, lpw, addr
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Full verification report.
    pub fn full_verification_report(&self) -> VerificationReport {
        self.verify()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FromReadComputer
// ═══════════════════════════════════════════════════════════════════════════

/// Computes the from-read (FR) relation from RF and CO.
#[derive(Debug, Clone)]
pub struct FromReadComputer {
    /// Events.
    events: Vec<Event>,
    /// Reads-from relation.
    rf: BitMatrix,
    /// Coherence order.
    co: BitMatrix,
    /// Computed from-read relation.
    fr: BitMatrix,
    /// Whether FR has been computed.
    computed: bool,
}

impl FromReadComputer {
    /// Create a new from-read computer.
    pub fn new(events: Vec<Event>, rf: BitMatrix, co: BitMatrix) -> Self {
        let n = events.len();
        Self { events, rf, co, fr: BitMatrix::new(n), computed: false }
    }

    /// Compute the from-read relation: fr = rf^{-1} ; co.
    pub fn compute_fr(&mut self) -> &BitMatrix {
        let rf_inv = self.rf.inverse();
        self.fr = rf_inv.compose(&self.co);
        self.computed = true;
        &self.fr
    }

    /// Compute FR from explicit RF and CO relations.
    pub fn compute_fr_from_rf_co(rf: &BitMatrix, co: &BitMatrix) -> BitMatrix {
        rf.inverse().compose(co)
    }

    /// Get all FR edges.
    pub fn fr_edges(&mut self) -> Vec<(EventId, EventId)> {
        if !self.computed { self.compute_fr(); }
        self.fr.edges()
    }

    /// Get immediate FR edges (not implied by transitivity).
    pub fn fr_immediate(&mut self) -> Vec<(EventId, EventId)> {
        if !self.computed { self.compute_fr(); }
        let mut immediate = Vec::new();
        let n = self.events.len();
        for (r, w) in self.fr.edges() {
            let mut is_immediate = true;
            for mid in 0..n {
                if mid != r && mid != w && self.fr.get(r, mid) && self.co.get(mid, w) {
                    is_immediate = false;
                    break;
                }
            }
            if is_immediate { immediate.push((r, w)); }
        }
        immediate
    }

    /// External FR: fr edges between different threads.
    pub fn external_fr(&mut self) -> Vec<(EventId, EventId)> {
        if !self.computed { self.compute_fr(); }
        self.fr.edges().into_iter()
            .filter(|&(r, w)| self.events[r].thread != self.events[w].thread)
            .collect()
    }

    /// Internal FR: fr edges within the same thread.
    pub fn internal_fr(&mut self) -> Vec<(EventId, EventId)> {
        if !self.computed { self.compute_fr(); }
        self.fr.edges().into_iter()
            .filter(|&(r, w)| self.events[r].thread == self.events[w].thread)
            .collect()
    }

    /// Validate the FR relation.
    pub fn validate_fr(&mut self) -> bool {
        if !self.computed { self.compute_fr(); }
        let n = self.events.len();
        for (r, w) in self.fr.edges() {
            if !self.events[r].is_read() { return false; }
            if !self.events[w].is_write() { return false; }
            if self.events[r].address != self.events[w].address { return false; }
        }
        true
    }

    /// Get the computed FR relation.
    pub fn get_fr(&self) -> &BitMatrix { &self.fr }
}

// ═══════════════════════════════════════════════════════════════════════════
// WriteCoalescing
// ═══════════════════════════════════════════════════════════════════════════

/// Result of write coalescing analysis.
#[derive(Debug, Clone)]
pub struct CoalescedWrite {
    /// Write events that are coalesced.
    pub write_ids: Vec<EventId>,
    /// Base address of the coalesced region.
    pub base_address: Address,
    /// Size of the coalesced region.
    pub size: usize,
    /// Threads involved.
    pub threads: Vec<ThreadId>,
}

/// Analyzer for write coalescing in GPU memory models.
#[derive(Debug, Clone)]
pub struct WriteCoalescingAnalyzer {
    /// Events.
    events: Vec<Event>,
    /// Coalescing window size (in addresses).
    window_size: u64,
    /// Detected coalesced writes.
    coalesced: Vec<CoalescedWrite>,
}

impl WriteCoalescingAnalyzer {
    /// Create a new analyzer.
    pub fn new(events: Vec<Event>, window_size: u64) -> Self {
        Self { events, window_size, coalesced: Vec::new() }
    }

    /// Detect coalesced writes.
    pub fn detect_coalescing(&mut self) -> &[CoalescedWrite] {
        self.coalesced.clear();
        // Group writes by warp (approximate by consecutive thread IDs)
        let mut writes_by_group: HashMap<ThreadId, Vec<(EventId, Address)>> = HashMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            if ev.is_write() {
                let warp_id = ev.thread / 32;
                writes_by_group.entry(warp_id).or_default().push((idx, ev.address));
            }
        }
        for (_warp, mut writes) in writes_by_group {
            writes.sort_by_key(|&(_, addr)| addr);
            let mut i = 0;
            while i < writes.len() {
                let base = writes[i].1;
                let mut group = vec![writes[i].0];
                let mut j = i + 1;
                while j < writes.len() && writes[j].1 - base < self.window_size {
                    group.push(writes[j].0);
                    j += 1;
                }
                if group.len() > 1 {
                    let threads: Vec<ThreadId> = group.iter()
                        .map(|&id| self.events[id].thread)
                        .collect::<HashSet<_>>().into_iter().collect();
                    self.coalesced.push(CoalescedWrite {
                        write_ids: group,
                        base_address: base,
                        size: (writes[j - 1].1 - base + 1) as usize,
                        threads,
                    });
                }
                i = j;
            }
        }
        &self.coalesced
    }

    /// Compute effective serialization under coalescing.
    pub fn effective_serialization(&self, co: &BitMatrix) -> BitMatrix {
        let n = self.events.len();
        let mut effective_co = co.clone();
        // Coalesced writes within the same transaction are unordered
        for cw in &self.coalesced {
            for i in 0..cw.write_ids.len() {
                for j in (i + 1)..cw.write_ids.len() {
                    effective_co.set(cw.write_ids[i], cw.write_ids[j], false);
                    effective_co.set(cw.write_ids[j], cw.write_ids[i], false);
                }
            }
        }
        effective_co
    }

    /// Get detected coalesced writes.
    pub fn get_coalesced(&self) -> &[CoalescedWrite] { &self.coalesced }

    /// Impact on race detection: coalesced writes are atomic wrt other threads.
    pub fn coalescing_impact_summary(&self) -> String {
        let total_writes = self.events.iter().filter(|e| e.is_write()).count();
        let coalesced_writes: usize = self.coalesced.iter().map(|c| c.write_ids.len()).sum();
        format!("Write coalescing: {}/{} writes coalesced into {} groups",
            coalesced_writes, total_writes, self.coalesced.len())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScopeAwareSerialization
// ═══════════════════════════════════════════════════════════════════════════

/// GPU scope for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SerializationScope {
    /// Per-thread coherence.
    Thread,
    /// Per-warp coherence.
    Warp,
    /// Per-CTA coherence.
    CTA,
    /// GPU-wide coherence.
    GPU,
    /// System-wide coherence.
    System,
}

impl fmt::Display for SerializationScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationScope::Thread => write!(f, "thread"),
            SerializationScope::Warp => write!(f, "warp"),
            SerializationScope::CTA => write!(f, "cta"),
            SerializationScope::GPU => write!(f, "gpu"),
            SerializationScope::System => write!(f, "system"),
        }
    }
}

/// Scope-aware write serialization for GPU memory models.
#[derive(Debug, Clone)]
pub struct ScopeAwareSerialization {
    /// Events.
    events: Vec<Event>,
    /// Per-scope coherence orders.
    per_scope_co: HashMap<SerializationScope, BitMatrix>,
    /// Thread-to-warp mapping.
    thread_warp: HashMap<ThreadId, u32>,
    /// Thread-to-CTA mapping.
    thread_cta: HashMap<ThreadId, u32>,
}

impl ScopeAwareSerialization {
    /// Create a new scope-aware serialization.
    pub fn new(events: Vec<Event>) -> Self {
        let n = events.len();
        let mut per_scope_co = HashMap::new();
        per_scope_co.insert(SerializationScope::Thread, BitMatrix::new(n));
        per_scope_co.insert(SerializationScope::Warp, BitMatrix::new(n));
        per_scope_co.insert(SerializationScope::CTA, BitMatrix::new(n));
        per_scope_co.insert(SerializationScope::GPU, BitMatrix::new(n));
        per_scope_co.insert(SerializationScope::System, BitMatrix::new(n));
        Self { events, per_scope_co, thread_warp: HashMap::new(), thread_cta: HashMap::new() }
    }

    /// Set thread mappings.
    pub fn set_thread_mapping(&mut self, warp: HashMap<ThreadId, u32>, cta: HashMap<ThreadId, u32>) {
        self.thread_warp = warp;
        self.thread_cta = cta;
    }

    /// Add a CO edge at a specific scope.
    pub fn add_co_edge(&mut self, from: EventId, to: EventId, scope: SerializationScope) {
        if let Some(mat) = self.per_scope_co.get_mut(&scope) {
            mat.set(from, to, true);
        }
        // Propagate to wider scopes
        let wider: Vec<SerializationScope> = [
            SerializationScope::Thread, SerializationScope::Warp,
            SerializationScope::CTA, SerializationScope::GPU,
            SerializationScope::System,
        ].iter().filter(|&&s| s > scope).copied().collect();
        for s in wider {
            if let Some(mat) = self.per_scope_co.get_mut(&s) {
                mat.set(from, to, true);
            }
        }
    }

    /// Get the CO relation at a specific scope.
    pub fn get_co_at_scope(&self, scope: SerializationScope) -> Option<&BitMatrix> {
        self.per_scope_co.get(&scope)
    }

    /// Compute scope-aware FR from RF and per-scope CO.
    pub fn compute_scope_fr(&self, rf: &BitMatrix, scope: SerializationScope) -> Option<BitMatrix> {
        self.per_scope_co.get(&scope).map(|co| rf.inverse().compose(co))
    }

    /// Check whether two threads are in the same scope.
    pub fn same_scope(&self, t1: ThreadId, t2: ThreadId, scope: SerializationScope) -> bool {
        match scope {
            SerializationScope::Thread => t1 == t2,
            SerializationScope::Warp => {
                self.thread_warp.get(&t1) == self.thread_warp.get(&t2)
            }
            SerializationScope::CTA => {
                self.thread_cta.get(&t1) == self.thread_cta.get(&t2)
            }
            SerializationScope::GPU | SerializationScope::System => true,
        }
    }

    /// Scope visibility: which writes are visible to a read at a given scope.
    pub fn visible_writes(&self, read_id: EventId, scope: SerializationScope) -> Vec<EventId> {
        let read_thread = self.events[read_id].thread;
        let addr = self.events[read_id].address;
        self.events.iter().enumerate()
            .filter(|(idx, ev)| {
                ev.is_write()
                    && ev.address == addr
                    && self.same_scope(ev.thread, read_thread, scope)
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Cross-scope serialization constraints.
    pub fn cross_scope_constraints(&self) -> Vec<(EventId, EventId, SerializationScope)> {
        let mut constraints = Vec::new();
        let scopes = [
            SerializationScope::CTA, SerializationScope::GPU, SerializationScope::System,
        ];
        for &scope in &scopes {
            if let Some(co) = self.per_scope_co.get(&scope) {
                for (from, to) in co.edges() {
                    // Check if this constraint is only at this scope level
                    let narrower = match scope {
                        SerializationScope::CTA => Some(SerializationScope::Warp),
                        SerializationScope::GPU => Some(SerializationScope::CTA),
                        SerializationScope::System => Some(SerializationScope::GPU),
                        _ => None,
                    };
                    let already_at_narrower = narrower.and_then(|ns| {
                        self.per_scope_co.get(&ns).map(|m| m.get(from, to))
                    }).unwrap_or(false);
                    if !already_at_narrower {
                        constraints.push((from, to, scope));
                    }
                }
            }
        }
        constraints
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SerializationOptimizer
// ═══════════════════════════════════════════════════════════════════════════

/// Optimizer for serialization checking.
#[derive(Debug, Clone)]
pub struct SerializationOptimizer {
    /// Constraint cache: known orderings.
    cache: HashMap<(EventId, EventId), bool>,
    /// Symmetry classes.
    symmetry_classes: Vec<Vec<EventId>>,
    /// Statistics.
    stats: OptimizerStats,
}

/// Statistics for the optimizer.
#[derive(Debug, Clone, Default)]
pub struct OptimizerStats {
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Number of symmetry reductions.
    pub symmetry_reductions: u64,
    /// Number of propagations.
    pub propagations: u64,
    /// Number of early terminations.
    pub early_terminations: u64,
}

impl SerializationOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            symmetry_classes: Vec::new(),
            stats: OptimizerStats::default(),
        }
    }

    /// Check ordering with caching.
    pub fn is_ordered(&mut self, a: EventId, b: EventId, co: &BitMatrix) -> bool {
        if let Some(&result) = self.cache.get(&(a, b)) {
            self.stats.cache_hits += 1;
            return result;
        }
        self.stats.cache_misses += 1;
        let result = co.get(a, b);
        self.cache.insert((a, b), result);
        result
    }

    /// Propagate constraints: if a < b and b < c, then a < c.
    pub fn propagate(&mut self, co: &BitMatrix) -> Vec<(EventId, EventId)> {
        let mut new_edges = Vec::new();
        let n = co.dim();
        for a in 0..n {
            for b in 0..n {
                if co.get(a, b) {
                    for c in 0..n {
                        if co.get(b, c) && !co.get(a, c) {
                            new_edges.push((a, c));
                            self.stats.propagations += 1;
                        }
                    }
                }
            }
        }
        new_edges
    }

    /// Detect symmetry classes among writes to the same address.
    pub fn detect_symmetry(&mut self, events: &[Event]) {
        self.symmetry_classes.clear();
        let mut by_profile: HashMap<(Address, ThreadId), Vec<EventId>> = HashMap::new();
        for (idx, ev) in events.iter().enumerate() {
            if ev.is_write() {
                by_profile.entry((ev.address, ev.thread)).or_default().push(idx);
            }
        }
        for (_, writes) in by_profile {
            if writes.len() > 1 {
                self.symmetry_classes.push(writes);
                self.stats.symmetry_reductions += 1;
            }
        }
    }

    /// Apply symmetry breaking: fix the first element of each symmetry class.
    pub fn symmetry_breaking_constraints(&self) -> Vec<(EventId, EventId)> {
        let mut constraints = Vec::new();
        for class in &self.symmetry_classes {
            for i in 1..class.len() {
                constraints.push((class[0], class[i]));
            }
        }
        constraints
    }

    /// Get statistics.
    pub fn stats(&self) -> &OptimizerStats { &self.stats }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SerializationOptimizer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_write_events(n: usize) -> Vec<Event> {
        (0..n).map(|i| Event {
            id: i,
            thread: i % 2,
            op_type: OpType::Write,
            address: 0x100,
            value: (i + 1) as u64,
            scope: Scope::System,
            po_index: i,
        }).collect()
    }

    #[test]
    fn test_serial_order() {
        let mut so = SerialOrder::new(0x100);
        so.append(0);
        so.append(1);
        so.append(2);
        assert!(so.is_before(0, 1));
        assert!(so.is_before(0, 2));
        assert!(!so.is_before(2, 0));
        assert_eq!(so.immediate_predecessor(1), Some(0));
        assert_eq!(so.immediate_successor(1), Some(2));
    }

    #[test]
    fn test_write_set() {
        let events = make_write_events(4);
        let exec = ExecutionGraph::new(events.clone());
        let ws = WriteSet::from_execution(&exec);
        assert_eq!(ws.len(), 4);
    }

    #[test]
    fn test_coherence_checker() {
        let events = make_write_events(3);
        let n = events.len();
        let po = BitMatrix::new(n);
        let rf = BitMatrix::new(n);
        let mut co = BitMatrix::new(n);
        co.set(0, 1, true);
        co.set(1, 2, true);
        co.set(0, 2, true);
        let mut checker = CoherenceChecker::new(events, po, rf, co);
        checker.check_coherence();
        assert!(checker.is_coherent());
    }

    #[test]
    fn test_from_read() {
        let n = 4;
        let mut rf = BitMatrix::new(n);
        let mut co = BitMatrix::new(n);
        rf.set(0, 1, true);
        co.set(0, 2, true);
        let fr = FromReadComputer::compute_fr_from_rf_co(&rf, &co);
        assert!(fr.get(1, 2));
    }
}
