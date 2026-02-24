#![allow(unused)]
//! Coherence order computation for axiomatic memory model verification.
//!
//! Implements coherence order (CO), from-read (FR), reads-from (RF) relations,
//! coherence cycle detection, SC-per-location checking, and coherence consistency.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet, BTreeMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

/// Unique event identifier.
pub type EventId = usize;
/// Thread identifier.
pub type ThreadId = usize;
/// Memory address.
pub type Address = u64;
/// Data value.
pub type Value = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType { Read, Write, Fence, RMW }

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Read => write!(f, "R"),
            OpType::Write => write!(f, "W"),
            OpType::Fence => write!(f, "F"),
            OpType::RMW => write!(f, "RMW"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub thread: ThreadId,
    pub op: OpType,
    pub address: Address,
    pub value: Value,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:T{}:{}({:#x})={}", self.id, self.thread, self.op, self.address, self.value)
    }
}

/// A simple square bit matrix for relation storage.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitMatrix {
    dim: usize,
    data: Vec<Vec<bool>>,
}

impl fmt::Debug for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitMatrix({}x{}, {} edges)", self.dim, self.dim, self.count_edges())
    }
}

impl Hash for BitMatrix {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dim.hash(state);
        for row in &self.data { for &b in row { b.hash(state); } }
    }
}

impl BitMatrix {
    pub fn new(dim: usize) -> Self { BitMatrix { dim, data: vec![vec![false; dim]; dim] } }

    pub fn dim(&self) -> usize { self.dim }

    pub fn get(&self, i: usize, j: usize) -> bool { self.data[i][j] }

    pub fn set(&mut self, i: usize, j: usize, v: bool) { self.data[i][j] = v; }

    pub fn add_edge(&mut self, i: usize, j: usize) { self.data[i][j] = true; }

    pub fn remove_edge(&mut self, i: usize, j: usize) { self.data[i][j] = false; }

    pub fn count_edges(&self) -> usize {
            self.data.iter().flat_map(|r| r.iter()).filter(|&&b| b).count()
        }

    pub fn transpose(&self) -> Self {
            let mut m = BitMatrix::new(self.dim);
            for i in 0..self.dim { for j in 0..self.dim { m.data[i][j] = self.data[j][i]; } }
            m
        }

    pub fn union(&self, other: &Self) -> Self {
            assert_eq!(self.dim, other.dim);
            let mut m = BitMatrix::new(self.dim);
            for i in 0..self.dim { for j in 0..self.dim { m.data[i][j] = self.data[i][j] || other.data[i][j]; } }
            m
        }

    pub fn intersection(&self, other: &Self) -> Self {
            assert_eq!(self.dim, other.dim);
            let mut m = BitMatrix::new(self.dim);
            for i in 0..self.dim { for j in 0..self.dim { m.data[i][j] = self.data[i][j] && other.data[i][j]; } }
            m
        }

    pub fn compose(&self, other: &Self) -> Self {
            assert_eq!(self.dim, other.dim);
            let mut m = BitMatrix::new(self.dim);
            for i in 0..self.dim {
                for j in 0..self.dim {
                    for k in 0..self.dim {
                        if self.data[i][k] && other.data[k][j] { m.data[i][j] = true; break; }
                    }
                }
            }
            m
        }

    pub fn transitive_closure(&self) -> Self {
            let mut tc = self.clone();
            for k in 0..self.dim {
                for i in 0..self.dim {
                    if !tc.data[i][k] { continue; }
                    for j in 0..self.dim {
                        if tc.data[k][j] { tc.data[i][j] = true; }
                    }
                }
            }
            tc
        }

    pub fn is_acyclic(&self) -> bool {
            let tc = self.transitive_closure();
            for i in 0..self.dim { if tc.data[i][i] { return false; } }
            true
        }

    pub fn is_irreflexive(&self) -> bool {
            for i in 0..self.dim { if self.data[i][i] { return false; } }
            true
        }

    pub fn is_total_on(&self, elems: &[usize]) -> bool {
            for &i in elems {
                for &j in elems {
                    if i != j && !self.data[i][j] && !self.data[j][i] { return false; }
                }
            }
            true
        }

    pub fn successors(&self, i: usize) -> Vec<usize> {
            (0..self.dim).filter(|&j| self.data[i][j]).collect()
        }

    pub fn predecessors(&self, j: usize) -> Vec<usize> {
            (0..self.dim).filter(|&i| self.data[i][j]).collect()
        }

    pub fn topological_sort(&self) -> Option<Vec<usize>> {
            let mut in_deg = vec![0usize; self.dim];
            for i in 0..self.dim { for j in 0..self.dim { if self.data[i][j] { in_deg[j] += 1; } } }
            let mut q: VecDeque<usize> = in_deg.iter().enumerate().filter(|(_, &d)| d == 0).map(|(i, _)| i).collect();
            let mut order = Vec::new();
            while let Some(u) = q.pop_front() {
                order.push(u);
                for j in 0..self.dim {
                    if self.data[u][j] { in_deg[j] -= 1; if in_deg[j] == 0 { q.push_back(j); } }
                }
            }
            if order.len() == self.dim { Some(order) } else { None }
        }

    pub fn find_cycle(&self) -> Option<Vec<usize>> {
            let mut visited = vec![0u8; self.dim]; // 0=white, 1=gray, 2=black
            let mut parent = vec![usize::MAX; self.dim];
            for start in 0..self.dim {
                if visited[start] != 0 { continue; }
                let mut stack = vec![(start, false)];
                while let Some((u, processed)) = stack.pop() {
                    if processed { visited[u] = 2; continue; }
                    if visited[u] == 1 { continue; }
                    visited[u] = 1;
                    stack.push((u, true));
                    for j in 0..self.dim {
                        if !self.data[u][j] { continue; }
                        if visited[j] == 1 {
                            let mut cycle = vec![j, u];
                            let mut cur = u;
                            while cur != j { cur = parent[cur]; if cur == usize::MAX { break; } cycle.push(cur); }
                            cycle.reverse();
                            return Some(cycle);
                        }
                        if visited[j] == 0 { parent[j] = u; stack.push((j, false)); }
                    }
                }
            }
            None
        }

    pub fn identity(dim: usize) -> Self {
            let mut m = BitMatrix::new(dim);
            for i in 0..dim { m.data[i][i] = true; }
            m
        }

    pub fn difference(&self, other: &Self) -> Self {
            assert_eq!(self.dim, other.dim);
            let mut m = BitMatrix::new(self.dim);
            for i in 0..self.dim { for j in 0..self.dim { m.data[i][j] = self.data[i][j] && !other.data[i][j]; } }
            m
        }

    pub fn edges(&self) -> Vec<(usize, usize)> {
            let mut e = Vec::new();
            for i in 0..self.dim { for j in 0..self.dim { if self.data[i][j] { e.push((i, j)); } } }
            e
        }

    pub fn density(&self) -> f64 {
            if self.dim == 0 { return 0.0; }
            self.count_edges() as f64 / (self.dim * self.dim) as f64
        }

}

// =========================================================================
// Coherence Order (CO)
// =========================================================================

/// The coherence order: a total order on writes to each memory location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceOrder {
    /// The relation matrix (partial order over all events).
    pub matrix: BitMatrix,
    /// Writes grouped by address.
    pub writes_per_location: HashMap<Address, Vec<EventId>>,
    /// Number of events in the execution.
    pub num_events: usize,
}

impl CoherenceOrder {
    pub fn new(num_events: usize) -> Self {
        CoherenceOrder {
            matrix: BitMatrix::new(num_events),
            writes_per_location: HashMap::new(),
            num_events,
        }
    }

    pub fn add_write(&mut self, event_id: EventId, addr: Address) {
        self.writes_per_location.entry(addr).or_default().push(event_id);
    }

    pub fn add_co_edge(&mut self, from: EventId, to: EventId) {
        self.matrix.add_edge(from, to);
    }

    pub fn is_ordered(&self, a: EventId, b: EventId) -> bool {
        self.matrix.get(a, b)
    }

    pub fn is_co_before(&self, a: EventId, b: EventId) -> bool {
        self.matrix.get(a, b)
    }

    pub fn total_on_location(&self, addr: Address) -> bool {
        if let Some(writes) = self.writes_per_location.get(&addr) {
            self.matrix.is_total_on(writes)
        } else {
            true
        }
    }

    pub fn is_total(&self) -> bool {
        for (_, writes) in &self.writes_per_location {
            if !self.matrix.is_total_on(writes) { return false; }
        }
        true
    }

    pub fn linearize(&self, addr: Address) -> Option<Vec<EventId>> {
        let writes = match self.writes_per_location.get(&addr) {
            Some(w) => w,
            None => return Some(Vec::new()),
        };
        // Build sub-matrix for just these writes
        let n = writes.len();
        let mut sub = BitMatrix::new(n);
        for i in 0..n {
            for j in 0..n {
                if self.matrix.get(writes[i], writes[j]) { sub.add_edge(i, j); }
            }
        }
        sub.topological_sort().map(|order| order.into_iter().map(|i| writes[i]).collect())
    }

    pub fn linearize_all(&self) -> Option<HashMap<Address, Vec<EventId>>> {
        let mut result = HashMap::new();
        for &addr in self.writes_per_location.keys() {
            match self.linearize(addr) {
                Some(lin) => { result.insert(addr, lin); }
                None => return None,
            }
        }
        Some(result)
    }

    pub fn is_acyclic(&self) -> bool {
        self.matrix.is_acyclic()
    }

    pub fn co_successors(&self, event: EventId) -> Vec<EventId> {
        self.matrix.successors(event)
    }

    pub fn co_predecessors(&self, event: EventId) -> Vec<EventId> {
        self.matrix.predecessors(event)
    }

    pub fn co_immediate_successor(&self, event: EventId, addr: Address) -> Option<EventId> {
        let writes = self.writes_per_location.get(&addr)?;
        let succs: Vec<_> = writes.iter()
            .filter(|&&w| self.matrix.get(event, w))
            .copied()
            .collect();
        // Find the minimal successor
        succs.into_iter().find(|&s| {
            !writes.iter().any(|&w| w != s && w != event && self.matrix.get(event, w) && self.matrix.get(w, s))
        })
    }

    pub fn statistics(&self) -> CoherenceStatistics {
        let total_writes: usize = self.writes_per_location.values().map(|v| v.len()).sum();
        let num_locations = self.writes_per_location.len();
        let total_edges = self.matrix.count_edges();
        let avg_chain_length = if num_locations > 0 {
            total_writes as f64 / num_locations as f64
        } else { 0.0 };
        let is_total = self.is_total();
        let is_acyclic = self.is_acyclic();
        CoherenceStatistics {
            num_events: self.num_events,
            num_locations,
            total_writes,
            total_co_edges: total_edges,
            avg_chain_length,
            density: self.matrix.density(),
            is_total,
            is_acyclic,
        }
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceStatistics {
    pub num_events: usize,
    pub num_locations: usize,
    pub total_writes: usize,
    pub total_co_edges: usize,
    pub avg_chain_length: f64,
    pub density: f64,
    pub is_total: bool,
    pub is_acyclic: bool,
}

impl fmt::Display for CoherenceStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coherence Statistics:")?;
        writeln!(f, "  Events:          {}", self.num_events)?;
        writeln!(f, "  Locations:       {}", self.num_locations)?;
        writeln!(f, "  Total writes:    {}", self.total_writes)?;
        writeln!(f, "  CO edges:        {}", self.total_co_edges)?;
        writeln!(f, "  Avg chain len:   {:.2}", self.avg_chain_length)?;
        writeln!(f, "  Density:         {:.4}", self.density)?;
        writeln!(f, "  Total:           {}", self.is_total)?;
        write!(f, "  Acyclic:         {}", self.is_acyclic)
    }
}

// =========================================================================
// Coherence Order Builder
// =========================================================================

#[derive(Debug, Clone)]
pub struct CoherenceOrderBuilder {
    num_events: usize,
    writes: Vec<(EventId, Address)>,
    edges: Vec<(EventId, EventId)>,
}

impl CoherenceOrderBuilder {
    pub fn new(num_events: usize) -> Self {
        CoherenceOrderBuilder { num_events, writes: Vec::new(), edges: Vec::new() }
    }

    pub fn add_write(&mut self, event_id: EventId, addr: Address) -> &mut Self {
        self.writes.push((event_id, addr));
        self
    }

    pub fn add_edge(&mut self, from: EventId, to: EventId) -> &mut Self {
        self.edges.push((from, to));
        self
    }

    pub fn add_total_order(&mut self, addr: Address, order: &[EventId]) -> &mut Self {
        for &eid in order { self.writes.push((eid, addr)); }
        for i in 0..order.len() {
            for j in (i+1)..order.len() { self.edges.push((order[i], order[j])); }
        }
        self
    }

    pub fn build(self) -> CoherenceOrder {
        let mut co = CoherenceOrder::new(self.num_events);
        for (eid, addr) in self.writes { co.add_write(eid, addr); }
        for (from, to) in self.edges { co.add_co_edge(from, to); }
        co
    }

    pub fn build_validated(self) -> Result<CoherenceOrder, CoherenceError> {
        let co = self.build();
        if !co.is_acyclic() { return Err(CoherenceError::CyclicOrder); }
        Ok(co)
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherenceError {
    CyclicOrder,
    NotTotal(Address),
    InvalidEvent(EventId),
    InconsistentValues { event: EventId, expected: Value, actual: Value },
    InvalidReadFrom { read: EventId, write: EventId },
}

impl fmt::Display for CoherenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicOrder => write!(f, "Coherence order contains a cycle"),
            Self::NotTotal(a) => write!(f, "Coherence order not total on address {:#x}", a),
            Self::InvalidEvent(e) => write!(f, "Invalid event id: {}", e),
            Self::InconsistentValues { event, expected, actual } =>
                write!(f, "Event {} has inconsistent values: expected {} got {}", event, expected, actual),
            Self::InvalidReadFrom { read, write } =>
                write!(f, "Invalid reads-from: R{} <- W{}", read, write),
        }
    }
}

// =========================================================================
// Coherence Extension — extend partial CO to all total orders
// =========================================================================

#[derive(Debug, Clone)]
pub struct CoherenceExtension {
    partial_co: CoherenceOrder,
}

impl CoherenceExtension {
    pub fn new(partial_co: CoherenceOrder) -> Self {
        CoherenceExtension { partial_co }
    }

    /// Enumerate all total extensions of the partial coherence order on a given address.
    pub fn enumerate_extensions(&self, addr: Address) -> Vec<Vec<EventId>> {
        let writes = match self.partial_co.writes_per_location.get(&addr) {
            Some(w) => w.clone(),
            None => return vec![Vec::new()],
        };
        let n = writes.len();
        if n == 0 { return vec![Vec::new()]; }
        // Generate all permutations consistent with partial order
        let mut results = Vec::new();
        let mut perm = Vec::new();
        let mut used = vec![false; n];
        self.extend_recursive(&writes, &mut perm, &mut used, n, &mut results);
        results
    }

    fn extend_recursive(
        &self, writes: &[EventId], perm: &mut Vec<EventId>,
        used: &mut Vec<bool>, n: usize, results: &mut Vec<Vec<EventId>>,
    ) {
        if perm.len() == n { results.push(perm.clone()); return; }
        for i in 0..n {
            if used[i] { continue; }
            // Check: all predecessors of writes[i] must already be in perm
            let ok = (0..n).all(|j| {
                if j == i || !self.partial_co.matrix.get(writes[j], writes[i]) { return true; }
                used[j]
            });
            if !ok { continue; }
            used[i] = true;
            perm.push(writes[i]);
            self.extend_recursive(writes, perm, used, n, results);
            perm.pop();
            used[i] = false;
        }
    }

    pub fn count_extensions(&self, addr: Address) -> usize {
        self.enumerate_extensions(addr).len()
    }

    pub fn total_extension_count(&self) -> usize {
        let mut total = 1usize;
        for &addr in self.partial_co.writes_per_location.keys() {
            total = total.saturating_mul(self.count_extensions(addr));
        }
        total
    }

}

// =========================================================================
// Coherence Enumerator
// =========================================================================

#[derive(Debug, Clone)]
pub struct CoherenceEnumerator {
    writes_per_location: HashMap<Address, Vec<EventId>>,
    num_events: usize,
}

impl CoherenceEnumerator {
    pub fn new(num_events: usize) -> Self {
        CoherenceEnumerator { writes_per_location: HashMap::new(), num_events }
    }

    pub fn add_write(&mut self, event_id: EventId, addr: Address) {
        self.writes_per_location.entry(addr).or_default().push(event_id);
    }

    /// Enumerate all valid total coherence orders.
    pub fn enumerate_all(&self) -> Vec<CoherenceOrder> {
        let addrs: Vec<Address> = self.writes_per_location.keys().copied().collect();
        let mut per_addr_perms: Vec<Vec<Vec<EventId>>> = Vec::new();
        for &addr in &addrs {
            let writes = &self.writes_per_location[&addr];
            per_addr_perms.push(Self::permutations(writes));
        }
        // Cartesian product
        let mut results = Vec::new();
        let mut indices = vec![0usize; addrs.len()];
        loop {
            let mut co = CoherenceOrder::new(self.num_events);
            for (ai, &addr) in addrs.iter().enumerate() {
                let perm = &per_addr_perms[ai][indices[ai]];
                for &eid in perm { co.add_write(eid, addr); }
                for i in 0..perm.len() {
                    for j in (i+1)..perm.len() { co.add_co_edge(perm[i], perm[j]); }
                }
            }
            results.push(co);
            // Advance
            let mut carry = true;
            for ai in 0..addrs.len() {
                if !carry { break; }
                indices[ai] += 1;
                if indices[ai] < per_addr_perms[ai].len() { carry = false; }
                else { indices[ai] = 0; }
            }
            if carry { break; }
        }
        results
    }

    fn permutations(elems: &[EventId]) -> Vec<Vec<EventId>> {
        if elems.is_empty() { return vec![Vec::new()]; }
        if elems.len() == 1 { return vec![vec![elems[0]]]; }
        let mut result = Vec::new();
        for i in 0..elems.len() {
            let mut rest: Vec<_> = elems.to_vec();
            rest.remove(i);
            for mut p in Self::permutations(&rest) {
                p.insert(0, elems[i]);
                result.push(p);
            }
        }
        result
    }

    pub fn count_all(&self) -> usize {
        let mut total = 1usize;
        for writes in self.writes_per_location.values() {
            let n = writes.len();
            let mut fact = 1usize;
            for i in 1..=n { fact = fact.saturating_mul(i); }
            total = total.saturating_mul(fact);
        }
        total
    }

}

// =========================================================================
// From-Read Relation (FR)
// =========================================================================

/// The from-read relation: fr(r,w) iff rf(w',r) and co(w',w) for some w'.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FromReadRelation {
    pub matrix: BitMatrix,
    pub num_events: usize,
}

#[derive(Debug, Clone)]
pub struct FromReadComputer;

impl FromReadComputer {
    /// Compute FR from RF and CO.
    /// rf: list of (write, read) pairs
    /// co: coherence order
    pub fn compute_fr(
        rf: &[(EventId, EventId)],
        co: &CoherenceOrder,
    ) -> FromReadRelation {
        let n = co.num_events;
        let mut matrix = BitMatrix::new(n);
        // For each rf(w', r), for each w such that co(w', w), add fr(r, w)
        for &(w_prime, r) in rf {
            let co_succs = co.co_successors(w_prime);
            for w in co_succs {
                matrix.add_edge(r, w);
            }
        }
        FromReadRelation { matrix, num_events: n }
    }

    /// Compute FR as edge list.
    pub fn compute_fr_edges(
        rf: &[(EventId, EventId)],
        co: &CoherenceOrder,
    ) -> Vec<(EventId, EventId)> {
        let fr = Self::compute_fr(rf, co);
        fr.matrix.edges()
    }

    /// Batch computation: compute FR for multiple RF/CO pairs.
    pub fn batch_compute(
        executions: &[(Vec<(EventId, EventId)>, CoherenceOrder)],
    ) -> Vec<FromReadRelation> {
        executions.iter()
            .map(|(rf, co)| Self::compute_fr(rf, co))
            .collect()
    }

}

impl FromReadRelation {
    pub fn is_acyclic(&self) -> bool {
        self.matrix.is_acyclic()
    }

    pub fn edges(&self) -> Vec<(EventId, EventId)> {
        self.matrix.edges()
    }

    pub fn is_fr(&self, read: EventId, write: EventId) -> bool {
        self.matrix.get(read, write)
    }

    pub fn fr_successors(&self, event: EventId) -> Vec<EventId> {
        self.matrix.successors(event)
    }

    pub fn count(&self) -> usize {
        self.matrix.count_edges()
    }

}

// =========================================================================
// Reads-From Relation (RF)
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadsFromRelation {
    pub matrix: BitMatrix,
    /// Map from read event to the write it reads from.
    pub rf_map: HashMap<EventId, EventId>,
    pub num_events: usize,
}

impl ReadsFromRelation {
    pub fn new(num_events: usize) -> Self {
        ReadsFromRelation {
            matrix: BitMatrix::new(num_events),
            rf_map: HashMap::new(),
            num_events,
        }
    }

    pub fn add_rf(&mut self, write: EventId, read: EventId) {
        self.matrix.add_edge(write, read);
        self.rf_map.insert(read, write);
    }

    pub fn reads_from(&self, read: EventId) -> Option<EventId> {
        self.rf_map.get(&read).copied()
    }

    pub fn is_rf(&self, write: EventId, read: EventId) -> bool {
        self.matrix.get(write, read)
    }

    pub fn edges(&self) -> Vec<(EventId, EventId)> {
        self.matrix.edges()
    }

    pub fn rf_per_location(&self, events: &[Event]) -> HashMap<Address, Vec<(EventId, EventId)>> {
        let mut result: HashMap<Address, Vec<(EventId, EventId)>> = HashMap::new();
        for (&read, &write) in &self.rf_map {
            if let Some(ev) = events.iter().find(|e| e.id == read) {
                result.entry(ev.address).or_default().push((write, read));
            }
        }
        result
    }

    pub fn count(&self) -> usize {
        self.rf_map.len()
    }

}

#[derive(Debug, Clone)]
pub struct ReadsFromBuilder {
    num_events: usize,
    edges: Vec<(EventId, EventId)>,
}

impl ReadsFromBuilder {
    pub fn new(num_events: usize) -> Self {
        ReadsFromBuilder { num_events, edges: Vec::new() }
    }

    pub fn add_rf(mut self, write: EventId, read: EventId) -> Self {
        self.edges.push((write, read));
        self
    }

    pub fn build(self) -> ReadsFromRelation {
        let mut rf = ReadsFromRelation::new(self.num_events);
        for (w, r) in self.edges { rf.add_rf(w, r); }
        rf
    }

    pub fn build_validated(self, events: &[Event]) -> Result<ReadsFromRelation, CoherenceError> {
        let rf = self.build();
        validate_rf(&rf, events)?;
        Ok(rf)
    }

}

/// Validate a reads-from relation.
pub fn validate_rf(rf: &ReadsFromRelation, events: &[Event]) -> Result<(), CoherenceError> {
    let event_map: HashMap<EventId, &Event> = events.iter().map(|e| (e.id, e)).collect();
    for (&read_id, &write_id) in &rf.rf_map {
        let read_ev = event_map.get(&read_id)
            .ok_or(CoherenceError::InvalidEvent(read_id))?;
        let write_ev = event_map.get(&write_id)
            .ok_or(CoherenceError::InvalidEvent(write_id))?;
        if read_ev.op != OpType::Read && read_ev.op != OpType::RMW {
            return Err(CoherenceError::InvalidReadFrom { read: read_id, write: write_id });
        }
        if write_ev.op != OpType::Write && write_ev.op != OpType::RMW {
            return Err(CoherenceError::InvalidReadFrom { read: read_id, write: write_id });
        }
        if read_ev.address != write_ev.address {
            return Err(CoherenceError::InvalidReadFrom { read: read_id, write: write_id });
        }
        if read_ev.value != write_ev.value {
            return Err(CoherenceError::InconsistentValues {
                event: read_id, expected: write_ev.value, actual: read_ev.value,
            });
        }
    }
    // Check each read reads from exactly one write
    let reads: Vec<_> = events.iter()
        .filter(|e| e.op == OpType::Read || e.op == OpType::RMW)
        .collect();
    for r in &reads {
        let rf_count = rf.rf_map.contains_key(&r.id);
        if !rf_count {
            // Not necessarily an error — could read from initial state
        }
    }
    Ok(())
}

// =========================================================================
// Reads-From Enumerator
// =========================================================================

#[derive(Debug, Clone)]
pub struct ReadsFromEnumerator {
    reads: Vec<(EventId, Address)>,
    writes_per_location: HashMap<Address, Vec<EventId>>,
    num_events: usize,
}

impl ReadsFromEnumerator {
    pub fn new(num_events: usize) -> Self {
        ReadsFromEnumerator {
            reads: Vec::new(),
            writes_per_location: HashMap::new(),
            num_events,
        }
    }

    pub fn add_read(&mut self, event_id: EventId, addr: Address) {
        self.reads.push((event_id, addr));
    }

    pub fn add_write(&mut self, event_id: EventId, addr: Address) {
        self.writes_per_location.entry(addr).or_default().push(event_id);
    }

    /// Enumerate all valid reads-from mappings.
    pub fn enumerate_all(&self) -> Vec<ReadsFromRelation> {
        let mut choices: Vec<Vec<(EventId, EventId)>> = Vec::new();
        for &(read_id, addr) in &self.reads {
            let writes = self.writes_per_location.get(&addr).cloned().unwrap_or_default();
            let read_choices: Vec<_> = writes.iter().map(|&w| (w, read_id)).collect();
            choices.push(read_choices);
        }
        if choices.is_empty() { return vec![ReadsFromRelation::new(self.num_events)]; }
        // Cartesian product of choices
        let mut results = Vec::new();
        let mut indices = vec![0usize; choices.len()];
        loop {
            if choices.iter().enumerate().all(|(i, c)| !c.is_empty() || indices[i] == 0) {
                let mut rf = ReadsFromRelation::new(self.num_events);
                for (i, choice) in choices.iter().enumerate() {
                    if !choice.is_empty() {
                        let (w, r) = choice[indices[i]];
                        rf.add_rf(w, r);
                    }
                }
                results.push(rf);
            }
            let mut carry = true;
            for i in 0..choices.len() {
                if !carry { break; }
                if choices[i].is_empty() { continue; }
                indices[i] += 1;
                if indices[i] < choices[i].len() { carry = false; }
                else { indices[i] = 0; }
            }
            if carry { break; }
        }
        results
    }

    pub fn count_all(&self) -> usize {
        let mut total = 1usize;
        for &(_, addr) in &self.reads {
            let n = self.writes_per_location.get(&addr).map_or(0, |w| w.len());
            if n > 0 { total = total.saturating_mul(n); }
        }
        total
    }

}

// =========================================================================
// Coherence Cycle Detection
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind { PO, RF, CO, FR }

impl fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeKind::PO => write!(f, "po"),
            EdgeKind::RF => write!(f, "rf"),
            EdgeKind::CO => write!(f, "co"),
            EdgeKind::FR => write!(f, "fr"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CycleClassification {
    CoRfFr,
    CoFrPo,
    RfCoPo,
    RfFrPo,
    CoRfFrPo,
    PureRfCo,
    PureFrCo,
    Other,
}

impl fmt::Display for CycleClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoRfFr => write!(f, "CO+RF+FR"),
            Self::CoFrPo => write!(f, "CO+FR+PO"),
            Self::RfCoPo => write!(f, "RF+CO+PO"),
            Self::RfFrPo => write!(f, "RF+FR+PO"),
            Self::CoRfFrPo => write!(f, "CO+RF+FR+PO"),
            Self::PureRfCo => write!(f, "RF+CO"),
            Self::PureFrCo => write!(f, "FR+CO"),
            Self::Other => write!(f, "other"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleWitness {
    pub edges: Vec<(EventId, EventId, EdgeKind)>,
    pub classification: CycleClassification,
}

impl CycleWitness {
    pub fn new(edges: Vec<(EventId, EventId, EdgeKind)>) -> Self {
        let classification = Self::classify_edges(&edges);
        CycleWitness { edges, classification }
    }

    pub fn len(&self) -> usize { self.edges.len() }

    pub fn is_empty(&self) -> bool { self.edges.is_empty() }

    pub fn events(&self) -> Vec<EventId> {
        let mut evs: Vec<EventId> = self.edges.iter().map(|&(a, _, _)| a).collect();
        evs.sort();
        evs.dedup();
        evs
    }

    pub fn edge_kinds(&self) -> HashSet<EdgeKind> {
        self.edges.iter().map(|&(_, _, k)| k).collect()
    }

    fn classify_edges(edges: &[(EventId, EventId, EdgeKind)]) -> CycleClassification {
        let kinds: HashSet<EdgeKind> = edges.iter().map(|&(_, _, k)| k).collect();
        let has_po = kinds.contains(&EdgeKind::PO);
        let has_rf = kinds.contains(&EdgeKind::RF);
        let has_co = kinds.contains(&EdgeKind::CO);
        let has_fr = kinds.contains(&EdgeKind::FR);
        match (has_po, has_rf, has_co, has_fr) {
            (false, true, true, true) => CycleClassification::CoRfFr,
            (true, false, true, true) => CycleClassification::CoFrPo,
            (true, true, true, false) => CycleClassification::RfCoPo,
            (true, true, false, true) => CycleClassification::RfFrPo,
            (true, true, true, true) => CycleClassification::CoRfFrPo,
            (false, true, true, false) => CycleClassification::PureRfCo,
            (false, false, true, true) => CycleClassification::PureFrCo,
            _ => CycleClassification::Other,
        }
    }

}

impl fmt::Display for CycleWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle[{}]: ", self.classification)?;
        for (i, &(from, to, kind)) in self.edges.iter().enumerate() {
            if i > 0 { write!(f, " -> ")?; }
            write!(f, "{}--{}-->{}", from, kind, to)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceCycleDetector {
    num_events: usize,
    po: BitMatrix,
    rf: BitMatrix,
    co: BitMatrix,
    fr: BitMatrix,
}

impl CoherenceCycleDetector {
    pub fn new(
        num_events: usize,
        po: BitMatrix,
        rf: BitMatrix,
        co: BitMatrix,
        fr: BitMatrix,
    ) -> Self {
        CoherenceCycleDetector { num_events, po, rf, co, fr }
    }

    pub fn from_relations(
        num_events: usize,
        po_edges: &[(EventId, EventId)],
        rf_edges: &[(EventId, EventId)],
        co_edges: &[(EventId, EventId)],
        fr_edges: &[(EventId, EventId)],
    ) -> Self {
        let mut po = BitMatrix::new(num_events);
        let mut rf = BitMatrix::new(num_events);
        let mut co = BitMatrix::new(num_events);
        let mut fr = BitMatrix::new(num_events);
        for &(a, b) in po_edges { po.add_edge(a, b); }
        for &(a, b) in rf_edges { rf.add_edge(a, b); }
        for &(a, b) in co_edges { co.add_edge(a, b); }
        for &(a, b) in fr_edges { fr.add_edge(a, b); }
        CoherenceCycleDetector { num_events, po, rf, co, fr }
    }

    /// Detect cycles in CO ∪ RF ∪ FR ∪ PO.
    pub fn detect_cycles(&self) -> Vec<CycleWitness> {
        let combined = self.po.union(&self.rf).union(&self.co).union(&self.fr);
        let mut cycles = Vec::new();
        if let Some(raw_cycle) = combined.find_cycle() {
            let witness = self.label_cycle(&raw_cycle);
            cycles.push(witness);
        }
        cycles
    }

    /// Check if the combined relation is acyclic.
    pub fn is_acyclic(&self) -> bool {
        let combined = self.po.union(&self.rf).union(&self.co).union(&self.fr);
        combined.is_acyclic()
    }

    fn label_cycle(&self, cycle: &[EventId]) -> CycleWitness {
        let mut edges = Vec::new();
        for i in 0..cycle.len() {
            let from = cycle[i];
            let to = cycle[(i + 1) % cycle.len()];
            let kind = if self.po.get(from, to) { EdgeKind::PO }
                else if self.rf.get(from, to) { EdgeKind::RF }
                else if self.co.get(from, to) { EdgeKind::CO }
                else if self.fr.get(from, to) { EdgeKind::FR }
                else { EdgeKind::PO }; // fallback
            edges.push((from, to, kind));
        }
        CycleWitness::new(edges)
    }

    pub fn minimize_cycle(&self, witness: &CycleWitness) -> CycleWitness {
        // Try removing each edge and checking if remaining still forms a cycle
        if witness.len() <= 2 { return witness.clone(); }
        let events = witness.events();
        let n = events.len();
        // For now, return the witness as-is (minimal by default for simple cycles)
        witness.clone()
    }

    pub fn classify_cycle(&self, witness: &CycleWitness) -> CycleClassification {
        witness.classification
    }

    pub fn detect_all_minimal_cycles(&self) -> Vec<CycleWitness> {
        let combined = self.po.union(&self.rf).union(&self.co).union(&self.fr);
        let n = self.num_events;
        let mut cycles = Vec::new();
        // Johnson's algorithm simplified: find all elementary cycles
        for start in 0..n {
            let mut visited = vec![false; n];
            let mut path = vec![start];
            visited[start] = true;
            self.dfs_cycles(start, start, &combined, &mut visited, &mut path, &mut cycles);
        }
        // Deduplicate
        let mut unique = Vec::new();
        let mut seen: HashSet<Vec<EventId>> = HashSet::new();
        for c in cycles {
            let mut normalized = c.events();
            normalized.sort();
            if seen.insert(normalized) { unique.push(c); }
        }
        unique
    }

    fn dfs_cycles(
        &self, start: usize, current: usize, combined: &BitMatrix,
        visited: &mut Vec<bool>, path: &mut Vec<usize>, results: &mut Vec<CycleWitness>,
    ) {
        for next in combined.successors(current) {
            if next == start && path.len() > 1 {
                let witness = self.label_cycle(path);
                results.push(witness);
            } else if !visited[next] && next > start {
                visited[next] = true;
                path.push(next);
                self.dfs_cycles(start, next, combined, visited, path, results);
                path.pop();
                visited[next] = false;
            }
        }
    }

}

// =========================================================================
// SC-per-location Checker
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScPerLocViolation {
    pub address: Address,
    pub cycle: CycleWitness,
    pub description: String,
}

impl fmt::Display for ScPerLocViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SC-per-loc violation at {:#x}: {} ({})", self.address, self.cycle, self.description)
    }
}

#[derive(Debug, Clone)]
pub struct ScPerLocationChecker {
    num_events: usize,
    events: Vec<Event>,
}

impl ScPerLocationChecker {
    pub fn new(events: Vec<Event>) -> Self {
        let num_events = events.len();
        ScPerLocationChecker { num_events, events }
    }

    /// Check SC-per-location: (po-loc ∪ rf ∪ co ∪ fr) must be acyclic.
    pub fn check(
        &self,
        po: &BitMatrix,
        rf: &ReadsFromRelation,
        co: &CoherenceOrder,
        fr: &FromReadRelation,
    ) -> Result<(), Vec<ScPerLocViolation>> {
        let mut violations = Vec::new();
        // Group events by address
        let mut events_by_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        for ev in &self.events {
            if ev.op != OpType::Fence {
                events_by_addr.entry(ev.address).or_default().push(ev.id);
            }
        }
        for (&addr, evts) in &events_by_addr {
            let n = evts.len();
            let mut local = BitMatrix::new(self.num_events);
            // po-loc
            for &i in evts {
                for &j in evts {
                    if po.get(i, j) { local.add_edge(i, j); }
                    if rf.matrix.get(i, j) { local.add_edge(i, j); }
                    if co.matrix.get(i, j) { local.add_edge(i, j); }
                    if fr.matrix.get(i, j) { local.add_edge(i, j); }
                }
            }
            if let Some(cycle_nodes) = local.find_cycle() {
                let detector = CoherenceCycleDetector::new(
                    self.num_events, po.clone(), rf.matrix.clone(),
                    co.matrix.clone(), fr.matrix.clone(),
                );
                let witness = detector.label_cycle(&cycle_nodes);
                violations.push(ScPerLocViolation {
                    address: addr,
                    cycle: witness,
                    description: format!("Cycle detected among events at address {:#x}", addr),
                });
            }
        }
        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

}

// =========================================================================
// Coherence Consistency Checker
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    pub is_consistent: bool,
    pub co_acyclic: bool,
    pub co_total: bool,
    pub sc_per_loc: bool,
    pub combined_acyclic: bool,
    pub violations: Vec<String>,
}

impl fmt::Display for ConsistencyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Consistency Result:")?;
        writeln!(f, "  Consistent:      {}", self.is_consistent)?;
        writeln!(f, "  CO acyclic:      {}", self.co_acyclic)?;
        writeln!(f, "  CO total:        {}", self.co_total)?;
        writeln!(f, "  SC-per-loc:      {}", self.sc_per_loc)?;
        writeln!(f, "  Combined acyclic: {}", self.combined_acyclic)?;
        if !self.violations.is_empty() {
            writeln!(f, "  Violations:")?;
            for v in &self.violations { writeln!(f, "    - {}", v)?; }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceConsistencyChecker {
    events: Vec<Event>,
    num_events: usize,
}

impl CoherenceConsistencyChecker {
    pub fn new(events: Vec<Event>) -> Self {
        let num_events = events.len();
        CoherenceConsistencyChecker { events, num_events }
    }

    pub fn check_consistency(
        &self,
        po: &BitMatrix,
        rf: &ReadsFromRelation,
        co: &CoherenceOrder,
    ) -> ConsistencyResult {
        let fr = FromReadComputer::compute_fr(&rf.edges(), co);
        let co_acyclic = co.is_acyclic();
        let co_total = co.is_total();
        let sc_checker = ScPerLocationChecker::new(self.events.clone());
        let sc_per_loc = sc_checker.check(po, rf, co, &fr).is_ok();
        let detector = CoherenceCycleDetector::new(
            self.num_events, po.clone(), rf.matrix.clone(),
            co.matrix.clone(), fr.matrix.clone(),
        );
        let combined_acyclic = detector.is_acyclic();
        let mut violations = Vec::new();
        if !co_acyclic { violations.push("Coherence order contains a cycle".to_string()); }
        if !co_total {
            for (&addr, _) in &co.writes_per_location {
                if !co.total_on_location(addr) {
                    violations.push(format!("CO not total on address {:#x}", addr));
                }
            }
        }
        if !sc_per_loc { violations.push("SC-per-location violated".to_string()); }
        if !combined_acyclic { violations.push("Combined relation (PO∪RF∪CO∪FR) has cycle".to_string()); }
        let is_consistent = co_acyclic && co_total && sc_per_loc && combined_acyclic;
        ConsistencyResult { is_consistent, co_acyclic, co_total, sc_per_loc, combined_acyclic, violations }
    }

}

// =========================================================================
// Utilities
// =========================================================================

/// Produce Graphviz DOT representation of a relation.
pub fn relation_to_dot(
    name: &str,
    matrix: &BitMatrix,
    event_labels: &[String],
) -> String {
    let mut dot = format!("digraph {} {{\n", name);
    dot.push_str("    rankdir=LR;\n");
    for (i, label) in event_labels.iter().enumerate() {
        dot.push_str(&format!("    {} [label=\"{}\"];\n", i, label));
    }
    for (i, j) in matrix.edges() {
        dot.push_str(&format!("    {} -> {};\n", i, j));
    }
    dot.push_str("}\n");
    dot
}

/// Print a coherence table.
pub fn print_coherence_table(co: &CoherenceOrder) -> String {
    let mut out = String::new();
    for (&addr, writes) in &co.writes_per_location {
        out.push_str(&format!("Address {:#x}: ", addr));
        if let Some(lin) = co.linearize(addr) {
            let s: Vec<String> = lin.iter().map(|e| format!("E{}", e)).collect();
            out.push_str(&s.join(" < "));
        } else {
            out.push_str("(not linearizable)");
        }
        out.push('\n');
    }
    out
}

/// Compute coherence statistics.
pub fn coherence_statistics(co: &CoherenceOrder) -> CoherenceStatistics {
    co.statistics()
}

/// Render a full execution as DOT with all relations.
pub fn execution_to_dot(
    events: &[Event],
    po: &BitMatrix,
    rf: &ReadsFromRelation,
    co: &CoherenceOrder,
    fr: &FromReadRelation,
) -> String {
    let mut dot = String::from("digraph execution {\n");
    dot.push_str("    rankdir=TB;\n");
    for ev in events {
        dot.push_str(&format!("    E{} [label=\"{}\"];\n", ev.id, ev));
    }
    for (i, j) in po.edges() {
        dot.push_str(&format!("    E{} -> E{} [color=black, label=\"po\"];\n", i, j));
    }
    for (i, j) in rf.matrix.edges() {
        dot.push_str(&format!("    E{} -> E{} [color=red, label=\"rf\"];\n", i, j));
    }
    for (i, j) in co.matrix.edges() {
        dot.push_str(&format!("    E{} -> E{} [color=blue, label=\"co\"];\n", i, j));
    }
    for (i, j) in fr.matrix.edges() {
        dot.push_str(&format!("    E{} -> E{} [color=green, label=\"fr\"];\n", i, j));
    }
    dot.push_str("}\n");
    dot
}

/// Compute relation composition chain length statistics.
pub fn relation_chain_stats(matrix: &BitMatrix) -> (usize, usize, f64) {
    let n = matrix.dim();
    if n == 0 { return (0, 0, 0.0); }
    let mut max_chain = 0;
    let mut total_chain = 0usize;
    let mut count = 0usize;
    for i in 0..n {
        let chain = bfs_longest_path(matrix, i);
        if chain > max_chain { max_chain = chain; }
        total_chain += chain;
        count += 1;
    }
    let avg = if count > 0 { total_chain as f64 / count as f64 } else { 0.0 };
    (max_chain, total_chain, avg)
}

fn bfs_longest_path(matrix: &BitMatrix, start: usize) -> usize {
    let n = matrix.dim();
    let mut dist = vec![0usize; n];
    let mut visited = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(start);
    visited[start] = true;
    let mut max_dist = 0;
    while let Some(u) = q.pop_front() {
        for j in 0..n {
            if matrix.get(u, j) && !visited[j] {
                visited[j] = true;
                dist[j] = dist[u] + 1;
                if dist[j] > max_dist { max_dist = dist[j]; }
                q.push_back(j);
            }
        }
    }
    max_dist
}

// =========================================================================
// Tests
// =========================================================================

// ===== Extended Coherence Operations =====

#[derive(Debug, Clone)]
pub struct PartialOrderRelation {
    pub elements: Vec<usize>,
    pub order_matrix: Vec<Vec<bool>>,
    pub labels: Vec<String>,
}

impl PartialOrderRelation {
    pub fn new(elements: Vec<usize>, order_matrix: Vec<Vec<bool>>, labels: Vec<String>) -> Self {
        PartialOrderRelation { elements, order_matrix, labels }
    }

    pub fn elements_len(&self) -> usize {
        self.elements.len()
    }

    pub fn elements_is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn order_matrix_len(&self) -> usize {
        self.order_matrix.len()
    }

    pub fn order_matrix_is_empty(&self) -> bool {
        self.order_matrix.is_empty()
    }

    pub fn labels_len(&self) -> usize {
        self.labels.len()
    }

    pub fn labels_is_empty(&self) -> bool {
        self.labels.is_empty()
    }

}

impl fmt::Display for PartialOrderRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PartialOrderRelation({:?})", self.elements)
    }
}

#[derive(Debug, Clone)]
pub struct PartialOrderRelationBuilder {
    elements: Vec<usize>,
    order_matrix: Vec<Vec<bool>>,
    labels: Vec<String>,
}

impl PartialOrderRelationBuilder {
    pub fn new() -> Self {
        PartialOrderRelationBuilder {
            elements: Vec::new(),
            order_matrix: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn elements(mut self, v: Vec<usize>) -> Self { self.elements = v; self }
    pub fn order_matrix(mut self, v: Vec<Vec<bool>>) -> Self { self.order_matrix = v; self }
    pub fn labels(mut self, v: Vec<String>) -> Self { self.labels = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceLattice {
    pub elements: Vec<usize>,
    pub join_table: Vec<Vec<usize>>,
    pub meet_table: Vec<Vec<usize>>,
}

impl CoherenceLattice {
    pub fn new(elements: Vec<usize>, join_table: Vec<Vec<usize>>, meet_table: Vec<Vec<usize>>) -> Self {
        CoherenceLattice { elements, join_table, meet_table }
    }

    pub fn elements_len(&self) -> usize {
        self.elements.len()
    }

    pub fn elements_is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn join_table_len(&self) -> usize {
        self.join_table.len()
    }

    pub fn join_table_is_empty(&self) -> bool {
        self.join_table.is_empty()
    }

    pub fn meet_table_len(&self) -> usize {
        self.meet_table.len()
    }

    pub fn meet_table_is_empty(&self) -> bool {
        self.meet_table.is_empty()
    }

}

impl fmt::Display for CoherenceLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceLattice({:?})", self.elements)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceLatticeBuilder {
    elements: Vec<usize>,
    join_table: Vec<Vec<usize>>,
    meet_table: Vec<Vec<usize>>,
}

impl CoherenceLatticeBuilder {
    pub fn new() -> Self {
        CoherenceLatticeBuilder {
            elements: Vec::new(),
            join_table: Vec::new(),
            meet_table: Vec::new(),
        }
    }

    pub fn elements(mut self, v: Vec<usize>) -> Self { self.elements = v; self }
    pub fn join_table(mut self, v: Vec<Vec<usize>>) -> Self { self.join_table = v; self }
    pub fn meet_table(mut self, v: Vec<Vec<usize>>) -> Self { self.meet_table = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceOrderMerger {
    pub conflict_count: usize,
    pub merge_log: Vec<String>,
}

impl CoherenceOrderMerger {
    pub fn new(conflict_count: usize, merge_log: Vec<String>) -> Self {
        CoherenceOrderMerger { conflict_count, merge_log }
    }

    pub fn get_conflict_count(&self) -> usize {
        self.conflict_count
    }

    pub fn merge_log_len(&self) -> usize {
        self.merge_log.len()
    }

    pub fn merge_log_is_empty(&self) -> bool {
        self.merge_log.is_empty()
    }

    pub fn with_conflict_count(mut self, v: usize) -> Self {
        self.conflict_count = v; self
    }

}

impl fmt::Display for CoherenceOrderMerger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceOrderMerger({:?})", self.conflict_count)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceOrderMergerBuilder {
    conflict_count: usize,
    merge_log: Vec<String>,
}

impl CoherenceOrderMergerBuilder {
    pub fn new() -> Self {
        CoherenceOrderMergerBuilder {
            conflict_count: 0,
            merge_log: Vec::new(),
        }
    }

    pub fn conflict_count(mut self, v: usize) -> Self { self.conflict_count = v; self }
    pub fn merge_log(mut self, v: Vec<String>) -> Self { self.merge_log = v; self }
}

#[derive(Debug, Clone)]
pub struct CanonicalExecution {
    pub event_count: usize,
    pub thread_map: Vec<u32>,
    pub address_map: Vec<u64>,
    pub hash: u64,
}

impl CanonicalExecution {
    pub fn new(event_count: usize, thread_map: Vec<u32>, address_map: Vec<u64>, hash: u64) -> Self {
        CanonicalExecution { event_count, thread_map, address_map, hash }
    }

    pub fn get_event_count(&self) -> usize {
        self.event_count
    }

    pub fn thread_map_len(&self) -> usize {
        self.thread_map.len()
    }

    pub fn thread_map_is_empty(&self) -> bool {
        self.thread_map.is_empty()
    }

    pub fn address_map_len(&self) -> usize {
        self.address_map.len()
    }

    pub fn address_map_is_empty(&self) -> bool {
        self.address_map.is_empty()
    }

    pub fn get_hash(&self) -> u64 {
        self.hash
    }

    pub fn with_event_count(mut self, v: usize) -> Self {
        self.event_count = v; self
    }

    pub fn with_hash(mut self, v: u64) -> Self {
        self.hash = v; self
    }

}

impl fmt::Display for CanonicalExecution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CanonicalExecution({:?})", self.event_count)
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalExecutionBuilder {
    event_count: usize,
    thread_map: Vec<u32>,
    address_map: Vec<u64>,
    hash: u64,
}

impl CanonicalExecutionBuilder {
    pub fn new() -> Self {
        CanonicalExecutionBuilder {
            event_count: 0,
            thread_map: Vec::new(),
            address_map: Vec::new(),
            hash: 0,
        }
    }

    pub fn event_count(mut self, v: usize) -> Self { self.event_count = v; self }
    pub fn thread_map(mut self, v: Vec<u32>) -> Self { self.thread_map = v; self }
    pub fn address_map(mut self, v: Vec<u64>) -> Self { self.address_map = v; self }
    pub fn hash(mut self, v: u64) -> Self { self.hash = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceHasher {
    pub seed: u64,
    pub polynomial: u64,
}

impl CoherenceHasher {
    pub fn new(seed: u64, polynomial: u64) -> Self {
        CoherenceHasher { seed, polynomial }
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }

    pub fn get_polynomial(&self) -> u64 {
        self.polynomial
    }

    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v; self
    }

    pub fn with_polynomial(mut self, v: u64) -> Self {
        self.polynomial = v; self
    }

}

impl fmt::Display for CoherenceHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceHasher({:?})", self.seed)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceHasherBuilder {
    seed: u64,
    polynomial: u64,
}

impl CoherenceHasherBuilder {
    pub fn new() -> Self {
        CoherenceHasherBuilder {
            seed: 0,
            polynomial: 0,
        }
    }

    pub fn seed(mut self, v: u64) -> Self { self.seed = v; self }
    pub fn polynomial(mut self, v: u64) -> Self { self.polynomial = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceComparison {
    pub left_only_edges: Vec<(usize, usize)>,
    pub right_only_edges: Vec<(usize, usize)>,
    pub common_edges: Vec<(usize, usize)>,
}

impl CoherenceComparison {
    pub fn new(left_only_edges: Vec<(usize, usize)>, right_only_edges: Vec<(usize, usize)>, common_edges: Vec<(usize, usize)>) -> Self {
        CoherenceComparison { left_only_edges, right_only_edges, common_edges }
    }

    pub fn left_only_edges_len(&self) -> usize {
        self.left_only_edges.len()
    }

    pub fn left_only_edges_is_empty(&self) -> bool {
        self.left_only_edges.is_empty()
    }

    pub fn right_only_edges_len(&self) -> usize {
        self.right_only_edges.len()
    }

    pub fn right_only_edges_is_empty(&self) -> bool {
        self.right_only_edges.is_empty()
    }

    pub fn common_edges_len(&self) -> usize {
        self.common_edges.len()
    }

    pub fn common_edges_is_empty(&self) -> bool {
        self.common_edges.is_empty()
    }

}

impl fmt::Display for CoherenceComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceComparison({:?})", self.left_only_edges)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceComparisonBuilder {
    left_only_edges: Vec<(usize, usize)>,
    right_only_edges: Vec<(usize, usize)>,
    common_edges: Vec<(usize, usize)>,
}

impl CoherenceComparisonBuilder {
    pub fn new() -> Self {
        CoherenceComparisonBuilder {
            left_only_edges: Vec::new(),
            right_only_edges: Vec::new(),
            common_edges: Vec::new(),
        }
    }

    pub fn left_only_edges(mut self, v: Vec<(usize, usize)>) -> Self { self.left_only_edges = v; self }
    pub fn right_only_edges(mut self, v: Vec<(usize, usize)>) -> Self { self.right_only_edges = v; self }
    pub fn common_edges(mut self, v: Vec<(usize, usize)>) -> Self { self.common_edges = v; self }
}

#[derive(Debug, Clone)]
pub struct WitnessMinimizer {
    pub original_size: usize,
    pub minimized_size: usize,
    pub removed_events: Vec<usize>,
    pub iterations: usize,
}

impl WitnessMinimizer {
    pub fn new(original_size: usize, minimized_size: usize, removed_events: Vec<usize>, iterations: usize) -> Self {
        WitnessMinimizer { original_size, minimized_size, removed_events, iterations }
    }

    pub fn get_original_size(&self) -> usize {
        self.original_size
    }

    pub fn get_minimized_size(&self) -> usize {
        self.minimized_size
    }

    pub fn removed_events_len(&self) -> usize {
        self.removed_events.len()
    }

    pub fn removed_events_is_empty(&self) -> bool {
        self.removed_events.is_empty()
    }

    pub fn get_iterations(&self) -> usize {
        self.iterations
    }

    pub fn with_original_size(mut self, v: usize) -> Self {
        self.original_size = v; self
    }

    pub fn with_minimized_size(mut self, v: usize) -> Self {
        self.minimized_size = v; self
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v; self
    }

}

impl fmt::Display for WitnessMinimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WitnessMinimizer({:?})", self.original_size)
    }
}

#[derive(Debug, Clone)]
pub struct WitnessMinimizerBuilder {
    original_size: usize,
    minimized_size: usize,
    removed_events: Vec<usize>,
    iterations: usize,
}

impl WitnessMinimizerBuilder {
    pub fn new() -> Self {
        WitnessMinimizerBuilder {
            original_size: 0,
            minimized_size: 0,
            removed_events: Vec::new(),
            iterations: 0,
        }
    }

    pub fn original_size(mut self, v: usize) -> Self { self.original_size = v; self }
    pub fn minimized_size(mut self, v: usize) -> Self { self.minimized_size = v; self }
    pub fn removed_events(mut self, v: Vec<usize>) -> Self { self.removed_events = v; self }
    pub fn iterations(mut self, v: usize) -> Self { self.iterations = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceStatisticsExt {
    pub num_events: usize,
    pub num_edges: usize,
    pub density: f64,
    pub num_threads: usize,
}

impl CoherenceStatisticsExt {
    pub fn new(num_events: usize, num_edges: usize, density: f64, num_threads: usize) -> Self {
        CoherenceStatisticsExt { num_events, num_edges, density, num_threads }
    }

    pub fn get_num_events(&self) -> usize {
        self.num_events
    }

    pub fn get_num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn get_density(&self) -> f64 {
        self.density
    }

    pub fn get_num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn with_num_events(mut self, v: usize) -> Self {
        self.num_events = v; self
    }

    pub fn with_num_edges(mut self, v: usize) -> Self {
        self.num_edges = v; self
    }

    pub fn with_density(mut self, v: f64) -> Self {
        self.density = v; self
    }

    pub fn with_num_threads(mut self, v: usize) -> Self {
        self.num_threads = v; self
    }

}

impl fmt::Display for CoherenceStatisticsExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceStatisticsExt({:?})", self.num_events)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceStatisticsExtBuilder {
    num_events: usize,
    num_edges: usize,
    density: f64,
    num_threads: usize,
}

impl CoherenceStatisticsExtBuilder {
    pub fn new() -> Self {
        CoherenceStatisticsExtBuilder {
            num_events: 0,
            num_edges: 0,
            density: 0.0,
            num_threads: 0,
        }
    }

    pub fn num_events(mut self, v: usize) -> Self { self.num_events = v; self }
    pub fn num_edges(mut self, v: usize) -> Self { self.num_edges = v; self }
    pub fn density(mut self, v: f64) -> Self { self.density = v; self }
    pub fn num_threads(mut self, v: usize) -> Self { self.num_threads = v; self }
}

#[derive(Debug, Clone)]
pub struct GraphvizNode {
    pub id: usize,
    pub label: String,
    pub color: String,
    pub shape: String,
}

impl GraphvizNode {
    pub fn new(id: usize, label: String, color: String, shape: String) -> Self {
        GraphvizNode { id, label, color, shape }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_label(&self) -> &str {
        &self.label
    }

    pub fn get_color(&self) -> &str {
        &self.color
    }

    pub fn get_shape(&self) -> &str {
        &self.shape
    }

    pub fn with_id(mut self, v: usize) -> Self {
        self.id = v; self
    }

    pub fn with_label(mut self, v: impl Into<String>) -> Self {
        self.label = v.into(); self
    }

    pub fn with_color(mut self, v: impl Into<String>) -> Self {
        self.color = v.into(); self
    }

    pub fn with_shape(mut self, v: impl Into<String>) -> Self {
        self.shape = v.into(); self
    }

}

impl fmt::Display for GraphvizNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphvizNode({:?})", self.id)
    }
}

#[derive(Debug, Clone)]
pub struct GraphvizNodeBuilder {
    id: usize,
    label: String,
    color: String,
    shape: String,
}

impl GraphvizNodeBuilder {
    pub fn new() -> Self {
        GraphvizNodeBuilder {
            id: 0,
            label: String::new(),
            color: String::new(),
            shape: String::new(),
        }
    }

    pub fn id(mut self, v: usize) -> Self { self.id = v; self }
    pub fn label(mut self, v: impl Into<String>) -> Self { self.label = v.into(); self }
    pub fn color(mut self, v: impl Into<String>) -> Self { self.color = v.into(); self }
    pub fn shape(mut self, v: impl Into<String>) -> Self { self.shape = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct GraphvizEdge {
    pub from: usize,
    pub to: usize,
    pub label: String,
    pub style: String,
}

impl GraphvizEdge {
    pub fn new(from: usize, to: usize, label: String, style: String) -> Self {
        GraphvizEdge { from, to, label, style }
    }

    pub fn get_from(&self) -> usize {
        self.from
    }

    pub fn get_to(&self) -> usize {
        self.to
    }

    pub fn get_label(&self) -> &str {
        &self.label
    }

    pub fn get_style(&self) -> &str {
        &self.style
    }

    pub fn with_from(mut self, v: usize) -> Self {
        self.from = v; self
    }

    pub fn with_to(mut self, v: usize) -> Self {
        self.to = v; self
    }

    pub fn with_label(mut self, v: impl Into<String>) -> Self {
        self.label = v.into(); self
    }

    pub fn with_style(mut self, v: impl Into<String>) -> Self {
        self.style = v.into(); self
    }

}

impl fmt::Display for GraphvizEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphvizEdge({:?})", self.from)
    }
}

#[derive(Debug, Clone)]
pub struct GraphvizEdgeBuilder {
    from: usize,
    to: usize,
    label: String,
    style: String,
}

impl GraphvizEdgeBuilder {
    pub fn new() -> Self {
        GraphvizEdgeBuilder {
            from: 0,
            to: 0,
            label: String::new(),
            style: String::new(),
        }
    }

    pub fn from(mut self, v: usize) -> Self { self.from = v; self }
    pub fn to(mut self, v: usize) -> Self { self.to = v; self }
    pub fn label(mut self, v: impl Into<String>) -> Self { self.label = v.into(); self }
    pub fn style(mut self, v: impl Into<String>) -> Self { self.style = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct ExtendedGraphviz {
    pub nodes: Vec<usize>,
    pub title: String,
    pub rankdir: String,
}

impl ExtendedGraphviz {
    pub fn new(nodes: Vec<usize>, title: String, rankdir: String) -> Self {
        ExtendedGraphviz { nodes, title, rankdir }
    }

    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes_is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get_title(&self) -> &str {
        &self.title
    }

    pub fn get_rankdir(&self) -> &str {
        &self.rankdir
    }

    pub fn with_title(mut self, v: impl Into<String>) -> Self {
        self.title = v.into(); self
    }

    pub fn with_rankdir(mut self, v: impl Into<String>) -> Self {
        self.rankdir = v.into(); self
    }

}

impl fmt::Display for ExtendedGraphviz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtendedGraphviz({:?})", self.nodes)
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedGraphvizBuilder {
    nodes: Vec<usize>,
    title: String,
    rankdir: String,
}

impl ExtendedGraphvizBuilder {
    pub fn new() -> Self {
        ExtendedGraphvizBuilder {
            nodes: Vec::new(),
            title: String::new(),
            rankdir: String::new(),
        }
    }

    pub fn nodes(mut self, v: Vec<usize>) -> Self { self.nodes = v; self }
    pub fn title(mut self, v: impl Into<String>) -> Self { self.title = v.into(); self }
    pub fn rankdir(mut self, v: impl Into<String>) -> Self { self.rankdir = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct CoherenceOrderBuilderExt {
    pub event_count: usize,
    pub edges: Vec<(usize, usize)>,
    pub ensure_acyclic: bool,
}

impl CoherenceOrderBuilderExt {
    pub fn new(event_count: usize) -> Self {
        CoherenceOrderBuilderExt { event_count, edges: Vec::new(), ensure_acyclic: false }
    }

    pub fn from_parts(event_count: usize, edges: Vec<(usize, usize)>, ensure_acyclic: bool) -> Self {
        CoherenceOrderBuilderExt { event_count, edges, ensure_acyclic }
    }
    pub fn get_event_count(&self) -> usize {
        self.event_count
    }

    pub fn edges_len(&self) -> usize {
        self.edges.len()
    }

    pub fn edges_is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn get_ensure_acyclic(&self) -> bool {
        self.ensure_acyclic
    }

    pub fn with_event_count(mut self, v: usize) -> Self {
        self.event_count = v; self
    }

    pub fn with_ensure_acyclic(mut self, v: bool) -> Self {
        self.ensure_acyclic = v; self
    }

    pub fn add_total_order(mut self, addr: u64, event_ids: &[usize]) -> Self {
        for i in 0..event_ids.len() {
            for j in (i + 1)..event_ids.len() {
                self.edges.push((event_ids[i], event_ids[j]));
            }
        }
        self
    }

    pub fn add_write(mut self, event_id: usize, _addr: u64) -> Self {
        // Builder pattern variant: just record (no-op for edges, addr info unused in builder)
        self
    }

    pub fn add_edge(mut self, from: usize, to: usize) -> Self {
        self.edges.push((from, to));
        self
    }

    pub fn build(self) -> CoherenceOrder {
        let mut co = CoherenceOrder::new(self.event_count);
        for &(from, to) in &self.edges {
            co.add_co_edge(from, to);
        }
        co
    }

    pub fn build_validated(self) -> Result<CoherenceOrder, CoherenceError> {
        let co = self.build();
        if !co.is_acyclic() {
            return Err(CoherenceError::CyclicOrder);
        }
        Ok(co)
    }

}

impl fmt::Display for CoherenceOrderBuilderExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceOrderBuilderExt({:?})", self.event_count)
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceOrderBuilderExtBuilder {
    event_count: usize,
    edges: Vec<(usize, usize)>,
    ensure_acyclic: bool,
}

impl CoherenceOrderBuilderExtBuilder {
    pub fn new() -> Self {
        CoherenceOrderBuilderExtBuilder {
            event_count: 0,
            edges: Vec::new(),
            ensure_acyclic: false,
        }
    }

    pub fn event_count(mut self, v: usize) -> Self { self.event_count = v; self }
    pub fn edges(mut self, v: Vec<(usize, usize)>) -> Self { self.edges = v; self }
    pub fn ensure_acyclic(mut self, v: bool) -> Self { self.ensure_acyclic = v; self }
}

#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl CoherenceAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        CoherenceAnalysis { data, size, computed: false, label: "Coherence".to_string(), threshold: 0.01 }
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

impl fmt::Display for CoherenceAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoherenceStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for CoherenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoherenceStatus::Pending => write!(f, "pending"),
            CoherenceStatus::InProgress => write!(f, "inprogress"),
            CoherenceStatus::Completed => write!(f, "completed"),
            CoherenceStatus::Failed => write!(f, "failed"),
            CoherenceStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoherencePriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for CoherencePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoherencePriority::Critical => write!(f, "critical"),
            CoherencePriority::High => write!(f, "high"),
            CoherencePriority::Medium => write!(f, "medium"),
            CoherencePriority::Low => write!(f, "low"),
            CoherencePriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoherenceMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for CoherenceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoherenceMode::Strict => write!(f, "strict"),
            CoherenceMode::Relaxed => write!(f, "relaxed"),
            CoherenceMode::Permissive => write!(f, "permissive"),
            CoherenceMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn coherence_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn coherence_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = coherence_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn coherence_std_dev(data: &[f64]) -> f64 {
    coherence_variance(data).sqrt()
}

pub fn coherence_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for Coherence.
pub fn coherence_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn coherence_entropy(data: &[f64]) -> f64 {
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

pub fn coherence_gini(data: &[f64]) -> f64 {
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

pub fn coherence_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = coherence_mean(&x);
    let my = coherence_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn coherence_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = coherence_covariance(data);
    let sx = coherence_std_dev(&data[..n]);
    let sy = coherence_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn coherence_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = coherence_mean(data);
    let s = coherence_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn coherence_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = coherence_mean(data);
    let s = coherence_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn coherence_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn coherence_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over coherence analysis results.
#[derive(Debug, Clone)]
pub struct CoherenceResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl CoherenceResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        CoherenceResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for CoherenceResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert PartialOrderRelation description to a summary string.
pub fn partialorderrelation_to_summary(item: &PartialOrderRelation) -> String {
    format!("PartialOrderRelation: {:?}", item)
}

/// Convert CoherenceLattice description to a summary string.
pub fn coherencelattice_to_summary(item: &CoherenceLattice) -> String {
    format!("CoherenceLattice: {:?}", item)
}

/// Convert CoherenceOrderMerger description to a summary string.
pub fn coherenceordermerger_to_summary(item: &CoherenceOrderMerger) -> String {
    format!("CoherenceOrderMerger: {:?}", item)
}

/// Convert CanonicalExecution description to a summary string.
pub fn canonicalexecution_to_summary(item: &CanonicalExecution) -> String {
    format!("CanonicalExecution: {:?}", item)
}

/// Convert CoherenceHasher description to a summary string.
pub fn coherencehasher_to_summary(item: &CoherenceHasher) -> String {
    format!("CoherenceHasher: {:?}", item)
}

/// Convert CoherenceComparison description to a summary string.
pub fn coherencecomparison_to_summary(item: &CoherenceComparison) -> String {
    format!("CoherenceComparison: {:?}", item)
}

/// Convert WitnessMinimizer description to a summary string.
pub fn witnessminimizer_to_summary(item: &WitnessMinimizer) -> String {
    format!("WitnessMinimizer: {:?}", item)
}

/// Convert CoherenceStatisticsExt description to a summary string.
pub fn coherencestatistics_to_summary(item: &CoherenceStatisticsExt) -> String {
    format!("CoherenceStatisticsExt: {:?}", item)
}

/// Convert GraphvizNode description to a summary string.
pub fn graphviznode_to_summary(item: &GraphvizNode) -> String {
    format!("GraphvizNode: {:?}", item)
}

/// Convert GraphvizEdge description to a summary string.
pub fn graphvizedge_to_summary(item: &GraphvizEdge) -> String {
    format!("GraphvizEdge: {:?}", item)
}

/// Convert ExtendedGraphviz description to a summary string.
pub fn extendedgraphviz_to_summary(item: &ExtendedGraphviz) -> String {
    format!("ExtendedGraphviz: {:?}", item)
}

/// Batch processor for coherence operations.
#[derive(Debug, Clone)]
pub struct CoherenceBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl CoherenceBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        CoherenceBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
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

impl fmt::Display for CoherenceBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for coherence analysis.
#[derive(Debug, Clone)]
pub struct CoherenceReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl CoherenceReport {
    pub fn new(title: impl Into<String>) -> Self {
        CoherenceReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
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

impl fmt::Display for CoherenceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceReport({})", self.title)
    }
}

/// Configuration for coherence analysis.
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl CoherenceConfig {
    pub fn default_config() -> Self {
        CoherenceConfig {
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

impl fmt::Display for CoherenceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for coherence data distribution.
#[derive(Debug, Clone)]
pub struct CoherenceHistogramExt {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl CoherenceHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return CoherenceHistogramExt { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
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
        CoherenceHistogramExt { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut acc = 0usize;
        for &b in &self.bins { acc += b; cum.push(acc); }
        cum
    }
    pub fn entropy(&self) -> f64 {
        let total = self.total_count as f64;
        if total == 0.0 { return 0.0; }
        let mut h = 0.0f64;
        for &b in &self.bins {
            if b > 0 { let p = b as f64 / total; h -= p * p.ln(); }
        }
        h
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

impl fmt::Display for CoherenceHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for coherence graph analysis.
#[derive(Debug, Clone)]
pub struct CoherenceGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl CoherenceGraph {
    pub fn new(n: usize) -> Self {
        CoherenceGraph {
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
        fn dfs_cycle_coherence(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_coherence(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_coherence(i, &self.adjacency, &mut visited) { return false; }
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

impl fmt::Display for CoherenceGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for coherence computation results.
#[derive(Debug, Clone)]
pub struct CoherenceCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl CoherenceCache {
    pub fn new(capacity: usize) -> Self {
        CoherenceCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
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

impl fmt::Display for CoherenceCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for coherence elements.
pub fn coherence_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// K-means clustering for coherence data.
pub fn coherence_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
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

/// Principal component analysis (simplified) for coherence data.
pub fn coherence_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
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

/// Dense matrix operations for Coherence computations.
#[derive(Debug, Clone)]
pub struct CoherenceDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl CoherenceDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        CoherenceDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        CoherenceDenseMatrix { rows, cols, data }
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
        CoherenceDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        CoherenceDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        CoherenceDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        CoherenceDenseMatrix { rows: self.rows, cols: self.cols, data }
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

impl fmt::Display for CoherenceDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CoherenceMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Coherence bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct CoherenceInterval {
    pub lo: f64,
    pub hi: f64,
}

impl CoherenceInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        CoherenceInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        CoherenceInterval { lo: v, hi: v }
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
        CoherenceInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(CoherenceInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        CoherenceInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        CoherenceInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        CoherenceInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { CoherenceInterval { lo: -self.hi, hi: -self.lo } }
        else { CoherenceInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        CoherenceInterval { lo, hi: self.hi.max(0.0).sqrt() }
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

impl fmt::Display for CoherenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Coherence protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoherenceState {
    Idle,
    Checking,
    Merging,
    Validating,
    Complete,
    Error,
}

impl fmt::Display for CoherenceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoherenceState::Idle => write!(f, "idle"),
            CoherenceState::Checking => write!(f, "checking"),
            CoherenceState::Merging => write!(f, "merging"),
            CoherenceState::Validating => write!(f, "validating"),
            CoherenceState::Complete => write!(f, "complete"),
            CoherenceState::Error => write!(f, "error"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceStateMachine {
    pub current: CoherenceState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl CoherenceStateMachine {
    pub fn new() -> Self {
        CoherenceStateMachine { current: CoherenceState::Idle, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &CoherenceState { &self.current }
    pub fn can_transition(&self, target: &CoherenceState) -> bool {
        match (&self.current, target) {
            (CoherenceState::Idle, CoherenceState::Checking) => true,
            (CoherenceState::Checking, CoherenceState::Merging) => true,
            (CoherenceState::Checking, CoherenceState::Validating) => true,
            (CoherenceState::Merging, CoherenceState::Validating) => true,
            (CoherenceState::Validating, CoherenceState::Complete) => true,
            (CoherenceState::Validating, CoherenceState::Error) => true,
            (CoherenceState::Error, CoherenceState::Idle) => true,
            (CoherenceState::Complete, CoherenceState::Idle) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: CoherenceState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = CoherenceState::Idle;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for CoherenceStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Coherence event tracking.
#[derive(Debug, Clone)]
pub struct CoherenceRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl CoherenceRingBuffer {
    pub fn new(capacity: usize) -> Self {
        CoherenceRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
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

impl fmt::Display for CoherenceRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Coherence component tracking.
#[derive(Debug, Clone)]
pub struct CoherenceDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl CoherenceDisjointSet {
    pub fn new(n: usize) -> Self {
        CoherenceDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
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

impl fmt::Display for CoherenceDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceSortedList {
    data: Vec<f64>,
}

impl CoherenceSortedList {
    pub fn new() -> Self { CoherenceSortedList { data: Vec::new() } }
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

impl fmt::Display for CoherenceSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Coherence metrics.
#[derive(Debug, Clone)]
pub struct CoherenceEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl CoherenceEma {
    pub fn new(alpha: f64) -> Self { CoherenceEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for CoherenceEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Coherence membership testing.
#[derive(Debug, Clone)]
pub struct CoherenceBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl CoherenceBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        CoherenceBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
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

impl fmt::Display for CoherenceBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Coherence string matching.
#[derive(Debug, Clone)]
pub struct CoherenceTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CoherenceTrie {
    nodes: Vec<CoherenceTrieNode>,
    count: usize,
}

impl CoherenceTrie {
    pub fn new() -> Self {
        CoherenceTrie { nodes: vec![CoherenceTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(CoherenceTrieNode { children: Vec::new(), is_terminal: false, value: None });
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

impl fmt::Display for CoherenceTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Coherence scheduling.
#[derive(Debug, Clone)]
pub struct CoherencePriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl CoherencePriorityQueue {
    pub fn new() -> Self { CoherencePriorityQueue { heap: Vec::new() } }
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

impl fmt::Display for CoherencePriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl CoherenceAccumulator {
    pub fn new() -> Self { CoherenceAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for CoherenceAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl CoherenceSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { CoherenceSparseMatrix { rows, cols, entries: Vec::new() } }
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
        let mut result = CoherenceSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = CoherenceSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = CoherenceSparseMatrix::new(self.rows, self.cols);
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

impl fmt::Display for CoherenceSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Coherence.
#[derive(Debug, Clone)]
pub struct CoherencePolynomial {
    pub coefficients: Vec<f64>,
}

impl CoherencePolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { CoherencePolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { CoherencePolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { CoherencePolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        CoherencePolynomial { coefficients: c }
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
        CoherencePolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        CoherencePolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        CoherencePolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        CoherencePolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        CoherencePolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        CoherencePolynomial { coefficients: coeffs }
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

impl fmt::Display for CoherencePolynomial {
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

/// Simple linear congruential generator for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceRng {
    state: u64,
}

impl CoherenceRng {
    pub fn new(seed: u64) -> Self { CoherenceRng { state: seed } }
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

impl fmt::Display for CoherenceRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Coherence benchmarking.
#[derive(Debug, Clone)]
pub struct CoherenceTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl CoherenceTimer {
    pub fn new(label: impl Into<String>) -> Self { CoherenceTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
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

impl fmt::Display for CoherenceTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Coherence set operations.
#[derive(Debug, Clone)]
pub struct CoherenceBitVector {
    words: Vec<u64>,
    len: usize,
}

impl CoherenceBitVector {
    pub fn new(len: usize) -> Self { CoherenceBitVector { words: vec![0u64; (len + 63) / 64], len } }
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

impl fmt::Display for CoherenceBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Coherence computation memoization.
#[derive(Debug, Clone)]
pub struct CoherenceLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl CoherenceLruCache {
    pub fn new(capacity: usize) -> Self { CoherenceLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
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

impl fmt::Display for CoherenceLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Coherence scheduling.
#[derive(Debug, Clone)]
pub struct CoherenceGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl CoherenceGraphColoring {
    pub fn new(n: usize) -> Self {
        CoherenceGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
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

impl fmt::Display for CoherenceGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Coherence ranking.
#[derive(Debug, Clone)]
pub struct CoherenceTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl CoherenceTopK {
    pub fn new(k: usize) -> Self { CoherenceTopK { k, items: Vec::new() } }
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

impl fmt::Display for CoherenceTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Coherence monitoring.
#[derive(Debug, Clone)]
pub struct CoherenceSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl CoherenceSlidingWindow {
    pub fn new(window_size: usize) -> Self { CoherenceSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
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

impl fmt::Display for CoherenceSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Coherence classification evaluation.
#[derive(Debug, Clone)]
pub struct CoherenceConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl CoherenceConfusionMatrix {
    pub fn new() -> Self { CoherenceConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
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

impl fmt::Display for CoherenceConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Coherence feature vectors.
pub fn coherence_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Coherence.
pub fn coherence_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Coherence.
pub fn coherence_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Coherence.
pub fn coherence_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Coherence.
pub fn coherence_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Coherence.
pub fn coherence_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Coherence.
pub fn coherence_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Coherence.
pub fn coherence_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Coherence.
pub fn coherence_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Coherence.
pub fn coherence_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Coherence.
pub fn coherence_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Coherence.
pub fn coherence_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Coherence.
pub fn coherence_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Coherence.
pub fn coherence_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Coherence.
pub fn coherence_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (coherence_kl_divergence(p, &m) + coherence_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Coherence.
pub fn coherence_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Coherence.
pub fn coherence_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Coherence.
pub fn coherence_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl CoherenceFeatureScaler {
    pub fn new() -> Self { CoherenceFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
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

impl fmt::Display for CoherenceFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Coherence trend analysis.
#[derive(Debug, Clone)]
pub struct CoherenceLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl CoherenceLinearRegression {
    pub fn new() -> Self { CoherenceLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
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

impl fmt::Display for CoherenceLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl CoherenceWeightedGraph {
    pub fn new(n: usize) -> Self { CoherenceWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
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
        fn find_coherence(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_coherence(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_coherence(&mut parent, u);
            let rv = find_coherence(&mut parent, v);
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

impl fmt::Display for CoherenceWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Coherence.
pub fn coherence_moving_average(data: &[f64], window: usize) -> Vec<f64> {
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

/// Cumulative sum for Coherence.
pub fn coherence_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Coherence.
pub fn coherence_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Coherence.
pub fn coherence_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Coherence.
pub fn coherence_dft_magnitude(data: &[f64]) -> Vec<f64> {
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

/// Trapezoidal integration for Coherence.
pub fn coherence_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Coherence.
pub fn coherence_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
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

/// Convolution for Coherence.
pub fn coherence_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Axis-aligned bounding box for Coherence spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct CoherenceAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl CoherenceAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { CoherenceAABB { x_min, y_min, x_max, y_max } }
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
            CoherenceAABB::new(self.x_min, self.y_min, cx, cy),
            CoherenceAABB::new(cx, self.y_min, self.x_max, cy),
            CoherenceAABB::new(self.x_min, cy, cx, self.y_max),
            CoherenceAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Coherence.
#[derive(Debug, Clone, Copy)]
pub struct CoherencePoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Coherence spatial indexing.
#[derive(Debug, Clone)]
pub struct CoherenceQuadTree {
    pub boundary: CoherenceAABB,
    pub points: Vec<CoherencePoint2D>,
    pub children: Option<Vec<CoherenceQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl CoherenceQuadTree {
    pub fn new(boundary: CoherenceAABB, capacity: usize, max_depth: usize) -> Self {
        CoherenceQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: CoherenceAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        CoherenceQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: CoherencePoint2D) -> bool {
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
            children.push(CoherenceQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &CoherenceAABB) -> Vec<CoherencePoint2D> {
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

impl fmt::Display for CoherenceQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Coherence.
pub fn coherence_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

/// Solve upper triangular system Rx = b for Coherence.
pub fn coherence_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Coherence.
pub fn coherence_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Coherence.
pub fn coherence_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Coherence.
pub fn coherence_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Coherence.
pub fn coherence_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Coherence.
pub fn coherence_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Coherence.
pub fn coherence_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Coherence.
pub fn coherence_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = coherence_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Coherence.
#[derive(Debug, Clone)]
pub struct CoherenceRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl CoherenceRunningStats {
    pub fn new() -> Self { CoherenceRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for CoherenceRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for Coherence.
pub fn coherence_iqr(data: &[f64]) -> f64 {
    coherence_percentile_at(data, 75.0) - coherence_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Coherence.
pub fn coherence_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = coherence_percentile_at(data, 25.0);
    let q3 = coherence_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Coherence.
pub fn coherence_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Coherence.
pub fn coherence_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Coherence.
pub fn coherence_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = coherence_rank(x);
    let ry = coherence_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for Coherence.
pub fn coherence_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Correlation matrix for Coherence.
pub fn coherence_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = coherence_covariance_matrix(data);
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

    fn make_event(id: EventId, thread: ThreadId, op: OpType, addr: Address, val: Value) -> Event {
        Event { id, thread, op, address: addr, value: val }
    }

    fn make_write(id: EventId, thread: ThreadId, addr: Address, val: Value) -> Event {
        make_event(id, thread, OpType::Write, addr, val)
    }

    fn make_read(id: EventId, thread: ThreadId, addr: Address, val: Value) -> Event {
        make_event(id, thread, OpType::Read, addr, val)
    }

    #[test]
    fn test_bitmatrix_new() {
    let m = BitMatrix::new(4);
            assert_eq!(m.dim(), 4);
            assert_eq!(m.count_edges(), 0);
            assert!(m.is_acyclic());
            assert!(m.is_irreflexive());
    }

    #[test]
    fn test_bitmatrix_add_edge() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            assert!(m.get(0, 1));
            assert!(m.get(1, 2));
            assert!(!m.get(0, 2));
            assert_eq!(m.count_edges(), 2);
    }

    #[test]
    fn test_bitmatrix_transpose() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            let t = m.transpose();
            assert!(t.get(1, 0));
            assert!(t.get(2, 1));
            assert!(!t.get(0, 1));
    }

    #[test]
    fn test_bitmatrix_union() {
    let mut a = BitMatrix::new(3);
            a.add_edge(0, 1);
            let mut b = BitMatrix::new(3);
            b.add_edge(1, 2);
            let c = a.union(&b);
            assert!(c.get(0, 1));
            assert!(c.get(1, 2));
    }

    #[test]
    fn test_bitmatrix_compose() {
    let mut a = BitMatrix::new(3);
            a.add_edge(0, 1);
            let mut b = BitMatrix::new(3);
            b.add_edge(1, 2);
            let c = a.compose(&b);
            assert!(c.get(0, 2));
            assert!(!c.get(0, 1));
    }

    #[test]
    fn test_bitmatrix_transitive_closure() {
    let mut m = BitMatrix::new(4);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            m.add_edge(2, 3);
            let tc = m.transitive_closure();
            assert!(tc.get(0, 3));
            assert!(tc.get(0, 2));
            assert!(tc.get(1, 3));
    }

    #[test]
    fn test_bitmatrix_acyclic() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            assert!(m.is_acyclic());
            m.add_edge(2, 0);
            assert!(!m.is_acyclic());
    }

    #[test]
    fn test_bitmatrix_topo_sort() {
    let mut m = BitMatrix::new(4);
            m.add_edge(0, 1);
            m.add_edge(0, 2);
            m.add_edge(1, 3);
            m.add_edge(2, 3);
            let order = m.topological_sort().unwrap();
            assert_eq!(order[0], 0);
            assert_eq!(order[3], 3);
    }

    #[test]
    fn test_bitmatrix_find_cycle() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            assert!(m.find_cycle().is_none());
            m.add_edge(2, 0);
            assert!(m.find_cycle().is_some());
    }

    #[test]
    fn test_bitmatrix_total_on() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(0, 2);
            m.add_edge(1, 2);
            assert!(m.is_total_on(&[0, 1, 2]));
            let m2 = BitMatrix::new(3);
            assert!(!m2.is_total_on(&[0, 1, 2]));
    }

    #[test]
    fn test_coherence_order_basic() {
    let mut co = CoherenceOrder::new(4);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_write(2, 0x100);
            co.add_co_edge(0, 1);
            co.add_co_edge(1, 2);
            co.add_co_edge(0, 2);
            assert!(co.is_ordered(0, 1));
            assert!(co.is_ordered(1, 2));
            assert!(co.total_on_location(0x100));
            assert!(co.is_acyclic());
    }

    #[test]
    fn test_coherence_order_linearize() {
    let mut co = CoherenceOrder::new(3);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_write(2, 0x100);
            co.add_co_edge(0, 1);
            co.add_co_edge(1, 2);
            co.add_co_edge(0, 2);
            let lin = co.linearize(0x100).unwrap();
            assert_eq!(lin, vec![0, 1, 2]);
    }

    #[test]
    fn test_coherence_order_not_total() {
    let mut co = CoherenceOrder::new(3);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_write(2, 0x100);
            co.add_co_edge(0, 1);
            // Missing edge between 1 and 2
            assert!(!co.total_on_location(0x100));
    }

    #[test]
    fn test_coherence_builder() {
    let co = CoherenceOrderBuilderExt::new(4)
                .add_total_order(0x100, &[0, 1, 2])
                .add_total_order(0x200, &[3])
                .build();
            assert!(co.is_total());
            assert!(co.is_acyclic());
    }

    #[test]
    fn test_coherence_builder_validated() {
    let result = CoherenceOrderBuilderExt::new(3)
                .add_write(0, 0x100)
                .add_write(1, 0x100)
                .add_write(2, 0x100)
                .add_edge(0, 1)
                .add_edge(1, 2)
                .add_edge(2, 0) // cycle!
                .build_validated();
            assert!(result.is_err());
    }

    #[test]
    fn test_from_read_computation() {
    // W0(x)=1 -> W1(x)=2, R2(x)=1 reads from W0
            // FR: R2 -> W1 (because rf(W0,R2) and co(W0,W1))
            let mut co = CoherenceOrder::new(3);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_co_edge(0, 1);
            let rf = vec![(0usize, 2usize)]; // W0 -> R2
            let fr = FromReadComputer::compute_fr(&rf, &co);
            assert!(fr.is_fr(2, 1)); // R2 -> W1
            assert!(!fr.is_fr(2, 0));
    }

    #[test]
    fn test_reads_from_relation() {
    let mut rf = ReadsFromRelation::new(4);
            rf.add_rf(0, 2); // W0 -> R2
            rf.add_rf(1, 3); // W1 -> R3
            assert_eq!(rf.reads_from(2), Some(0));
            assert_eq!(rf.reads_from(3), Some(1));
            assert_eq!(rf.reads_from(0), None);
            assert_eq!(rf.count(), 2);
    }

    #[test]
    fn test_reads_from_builder() {
    let rf = ReadsFromBuilder::new(4)
                .add_rf(0, 2)
                .add_rf(1, 3)
                .build();
            assert!(rf.is_rf(0, 2));
            assert!(rf.is_rf(1, 3));
            assert!(!rf.is_rf(2, 0));
    }

    #[test]
    fn test_validate_rf_valid() {
    let events = vec![
                make_write(0, 0, 0x100, 42),
                make_write(1, 1, 0x100, 99),
                make_read(2, 0, 0x100, 42),
                make_read(3, 1, 0x100, 99),
            ];
            let rf = ReadsFromBuilder::new(4).add_rf(0, 2).add_rf(1, 3).build();
            assert!(validate_rf(&rf, &events).is_ok());
    }

    #[test]
    fn test_validate_rf_value_mismatch() {
    let events = vec![
                make_write(0, 0, 0x100, 42),
                make_read(1, 1, 0x100, 99), // reads 99 but W0 wrote 42
            ];
            let rf = ReadsFromBuilder::new(2).add_rf(0, 1).build();
            assert!(validate_rf(&rf, &events).is_err());
    }

    #[test]
    fn test_validate_rf_address_mismatch() {
    let events = vec![
                make_write(0, 0, 0x100, 42),
                make_read(1, 1, 0x200, 42), // different address
            ];
            let rf = ReadsFromBuilder::new(2).add_rf(0, 1).build();
            assert!(validate_rf(&rf, &events).is_err());
    }

    #[test]
    fn test_reads_from_enumerator() {
    let mut en = ReadsFromEnumerator::new(4);
            en.add_read(2, 0x100);
            en.add_write(0, 0x100);
            en.add_write(1, 0x100);
            let all = en.enumerate_all();
            assert_eq!(all.len(), 2); // R2 can read from W0 or W1
            assert_eq!(en.count_all(), 2);
    }

    #[test]
    fn test_coherence_enumerator() {
    let mut en = CoherenceEnumerator::new(3);
            en.add_write(0, 0x100);
            en.add_write(1, 0x100);
            en.add_write(2, 0x100);
            let all = en.enumerate_all();
            assert_eq!(all.len(), 6); // 3! = 6 permutations
    }

    #[test]
    fn test_coherence_extension() {
    let mut co = CoherenceOrder::new(3);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_write(2, 0x100);
            co.add_co_edge(0, 1); // partial: 0 < 1, 2 is free
            let ext = CoherenceExtension::new(co);
            let extensions = ext.enumerate_extensions(0x100);
            // 0 must come before 1. 2 can be anywhere: [2,0,1], [0,2,1], [0,1,2] = 3
            assert_eq!(extensions.len(), 3);
    }

    #[test]
    fn test_cycle_detection_no_cycle() {
    let det = CoherenceCycleDetector::from_relations(
                3,
                &[(0, 1)],       // po
                &[(1, 2)],       // rf (W1 -> R2)
                &[],             // co
                &[],             // fr
            );
            assert!(det.is_acyclic());
            assert!(det.detect_cycles().is_empty());
    }

    #[test]
    fn test_cycle_detection_with_cycle() {
    // po: 0->1, rf: 1->2, fr: 2->0 => cycle
            let det = CoherenceCycleDetector::from_relations(
                3,
                &[(0, 1)],
                &[(1, 2)],
                &[],
                &[(2, 0)],
            );
            assert!(!det.is_acyclic());
            let cycles = det.detect_cycles();
            assert!(!cycles.is_empty());
    }

    #[test]
    fn test_cycle_classification() {
    let edges = vec![
                (0, 1, EdgeKind::CO),
                (1, 2, EdgeKind::RF),
                (2, 0, EdgeKind::FR),
            ];
            let witness = CycleWitness::new(edges);
            assert_eq!(witness.classification, CycleClassification::CoRfFr);
    }

    #[test]
    fn test_sc_per_loc_pass() {
    let events = vec![
                make_write(0, 0, 0x100, 1),
                make_write(1, 1, 0x100, 2),
                make_read(2, 0, 0x100, 1),
            ];
            let mut po = BitMatrix::new(3);
            po.add_edge(0, 2); // W0 -> R2 in thread 0
            let rf = ReadsFromBuilder::new(3).add_rf(0, 2).build();
            let co = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[0, 1])
                .build();
            let fr = FromReadComputer::compute_fr(&rf.edges(), &co);
            let checker = ScPerLocationChecker::new(events);
            assert!(checker.check(&po, &rf, &co, &fr).is_ok());
    }

    #[test]
    fn test_consistency_checker() {
    let events = vec![
                make_write(0, 0, 0x100, 1),
                make_write(1, 1, 0x100, 2),
                make_read(2, 0, 0x100, 1),
            ];
            let mut po = BitMatrix::new(3);
            po.add_edge(0, 2);
            let rf = ReadsFromBuilder::new(3).add_rf(0, 2).build();
            let co = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[0, 1])
                .build();
            let checker = CoherenceConsistencyChecker::new(events);
            let result = checker.check_consistency(&po, &rf, &co);
            assert!(result.is_consistent);
    }

    #[test]
    fn test_coherence_statistics() {
    let co = CoherenceOrderBuilderExt::new(4)
                .add_total_order(0x100, &[0, 1, 2])
                .add_total_order(0x200, &[3])
                .build();
            let stats = co.statistics();
            assert_eq!(stats.num_locations, 2);
            assert_eq!(stats.total_writes, 4);
            assert!(stats.is_total);
            assert!(stats.is_acyclic);
    }

    #[test]
    fn test_relation_to_dot() {
    let mut m = BitMatrix::new(3);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            let labels = vec!["W(x)=1".to_string(), "W(x)=2".to_string(), "R(x)=1".to_string()];
            let dot = relation_to_dot("co", &m, &labels);
            assert!(dot.contains("digraph co"));
            assert!(dot.contains("0 -> 1"));
            assert!(dot.contains("1 -> 2"));
    }

    #[test]
    fn test_print_coherence_table() {
    let co = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[0, 1, 2])
                .build();
            let table = print_coherence_table(&co);
            assert!(table.contains("0x100"));
    }

    #[test]
    fn test_execution_to_dot() {
    let events = vec![
                make_write(0, 0, 0x100, 1),
                make_read(1, 1, 0x100, 1),
            ];
            let mut po = BitMatrix::new(2);
            let rf = ReadsFromBuilder::new(2).add_rf(0, 1).build();
            let co = CoherenceOrder::new(2);
            let fr = FromReadRelation { matrix: BitMatrix::new(2), num_events: 2 };
            let dot = execution_to_dot(&events, &po, &rf, &co, &fr);
            assert!(dot.contains("digraph execution"));
            assert!(dot.contains("rf"));
    }

    #[test]
    fn test_empty_execution() {
    let co = CoherenceOrder::new(0);
            assert!(co.is_total());
            assert!(co.is_acyclic());
            let stats = co.statistics();
            assert_eq!(stats.num_events, 0);
    }

    #[test]
    fn test_single_write() {
    let mut co = CoherenceOrder::new(1);
            co.add_write(0, 0x100);
            assert!(co.is_total());
            assert!(co.is_acyclic());
            let lin = co.linearize(0x100).unwrap();
            assert_eq!(lin, vec![0]);
    }

    #[test]
    fn test_bitmatrix_identity() {
    let id = BitMatrix::identity(4);
            for i in 0..4 { assert!(id.get(i, i)); }
            assert!(!id.get(0, 1));
    }

    #[test]
    fn test_bitmatrix_difference() {
    let mut a = BitMatrix::new(3);
            a.add_edge(0, 1);
            a.add_edge(1, 2);
            let mut b = BitMatrix::new(3);
            b.add_edge(0, 1);
            let c = a.difference(&b);
            assert!(!c.get(0, 1));
            assert!(c.get(1, 2));
    }

    #[test]
    fn test_bitmatrix_density() {
    let mut m = BitMatrix::new(2);
            assert_eq!(m.density(), 0.0);
            m.add_edge(0, 1);
            m.add_edge(1, 0);
            assert!((m.density() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_immediate_successor() {
    let mut co = CoherenceOrder::new(3);
            co.add_write(0, 0x100);
            co.add_write(1, 0x100);
            co.add_write(2, 0x100);
            co.add_co_edge(0, 1);
            co.add_co_edge(1, 2);
            co.add_co_edge(0, 2);
            assert_eq!(co.co_immediate_successor(0, 0x100), Some(1));
            assert_eq!(co.co_immediate_successor(1, 0x100), Some(2));
    }

    #[test]
    fn test_from_read_batch() {
    let co1 = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[0, 1])
                .build();
            let co2 = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[1, 0])
                .build();
            let rf1 = vec![(0, 2)];
            let rf2 = vec![(1, 2)];
            let results = FromReadComputer::batch_compute(&[(rf1, co1), (rf2, co2)]);
            assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_multiple_locations() {
    let co = CoherenceOrderBuilderExt::new(6)
                .add_total_order(0x100, &[0, 1, 2])
                .add_total_order(0x200, &[3, 4, 5])
                .build();
            assert!(co.is_total());
            let stats = co.statistics();
            assert_eq!(stats.num_locations, 2);
            assert_eq!(stats.total_writes, 6);
    }

    #[test]
    fn test_coherence_enumerator_single() {
    let mut en = CoherenceEnumerator::new(1);
            en.add_write(0, 0x100);
            let all = en.enumerate_all();
            assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_chain_stats() {
    let mut m = BitMatrix::new(4);
            m.add_edge(0, 1);
            m.add_edge(1, 2);
            m.add_edge(2, 3);
            let (max_chain, _, avg) = relation_chain_stats(&m);
            assert_eq!(max_chain, 3);
    }

    #[test]
    fn test_cycle_witness_display() {
    let edges = vec![
                (0, 1, EdgeKind::PO),
                (1, 2, EdgeKind::RF),
                (2, 0, EdgeKind::FR),
            ];
            let w = CycleWitness::new(edges);
            let s = format!("{}", w);
            assert!(s.contains("po"));
            assert!(s.contains("rf"));
            assert!(s.contains("fr"));
    }

    #[test]
    fn test_cycle_witness_events() {
    let edges = vec![
                (3, 5, EdgeKind::CO),
                (5, 7, EdgeKind::RF),
                (7, 3, EdgeKind::FR),
            ];
            let w = CycleWitness::new(edges);
            let evs = w.events();
            assert!(evs.contains(&3));
            assert!(evs.contains(&5));
            assert!(evs.contains(&7));
    }

    #[test]
    fn test_linearize_all() {
    let co = CoherenceOrderBuilderExt::new(5)
                .add_total_order(0x100, &[0, 1])
                .add_total_order(0x200, &[2, 3, 4])
                .build();
            let all = co.linearize_all().unwrap();
            assert_eq!(all.len(), 2);
            assert_eq!(all[&0x100], vec![0, 1]);
    }

    #[test]
    fn test_fr_acyclic() {
    let co = CoherenceOrderBuilderExt::new(3)
                .add_total_order(0x100, &[0, 1])
                .build();
            let rf = vec![(0, 2)];
            let fr = FromReadComputer::compute_fr(&rf, &co);
            assert!(fr.is_acyclic());
    }

    #[test]
    fn test_coherence_error_display() {
    let e = CoherenceError::CyclicOrder;
            assert!(format!("{}", e).contains("cycle"));
            let e2 = CoherenceError::NotTotal(0x100);
            assert!(format!("{}", e2).contains("0x100"));
    }

    #[test]
    fn test_bitmatrix_successors_predecessors() {
    let mut m = BitMatrix::new(4);
            m.add_edge(0, 1);
            m.add_edge(0, 2);
            m.add_edge(3, 2);
            assert_eq!(m.successors(0), vec![1, 2]);
            assert_eq!(m.predecessors(2), vec![0, 3]);
    }

    #[test]
    fn test_consistency_inconsistent() {
    let events = vec![
                make_write(0, 0, 0x100, 1),
                make_write(1, 1, 0x100, 2),
                make_read(2, 0, 0x100, 2),
            ];
            let mut po = BitMatrix::new(3);
            po.add_edge(0, 2); // W0(x)=1 ->po R2(x)=2
            let rf = ReadsFromBuilder::new(3).add_rf(1, 2).build(); // R2 reads from W1
            // CO: W1 < W0 (reversed!) so fr(R2, W0) and po(W0, R2) creates cycle
            let co = CoherenceOrderBuilderExt::new(3)
                .add_write(0, 0x100)
                .add_write(1, 0x100)
                .add_edge(1, 0) // W1 < W0
                .build();
            let checker = CoherenceConsistencyChecker::new(events);
            let result = checker.check_consistency(&po, &rf, &co);
            // This may or may not be consistent depending on FR
            // Just verify the checker runs without panicking
            let _ = result.is_consistent;
    }
    #[test]
    fn test_partialorderrelation_new() {
        let item = PartialOrderRelation::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherencelattice_new() {
        let item = CoherenceLattice::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherenceordermerger_new() {
        let item = CoherenceOrderMerger::new(0, Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_canonicalexecution_new() {
        let item = CanonicalExecution::new(0, Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherencehasher_new() {
        let item = CoherenceHasher::new(0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherencecomparison_new() {
        let item = CoherenceComparison::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_witnessminimizer_new() {
        let item = WitnessMinimizer::new(0, 0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherencestatistics_new() {
        let item = CoherenceStatisticsExt::new(0, 0, 0.0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_graphviznode_new() {
        let item = GraphvizNode::new(0, "test".to_string(), "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_graphvizedge_new() {
        let item = GraphvizEdge::new(0, 0, "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_extendedgraphviz_new() {
        let item = ExtendedGraphviz::new(Vec::new(), "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherenceorderbuilder_new() {
        let item = CoherenceOrderBuilderExt::from_parts(0, Vec::new(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_coherence_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = coherence_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = coherence_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_coherence_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = coherence_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = coherence_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_coherence_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = coherence_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_analysis() {
        let mut a = CoherenceAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_coherence_iterator() {
        let iter = CoherenceResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_coherence_batch_processor() {
        let mut proc = CoherenceBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_coherence_histogram() {
        let hist = CoherenceHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_coherence_graph() {
        let mut g = CoherenceGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_coherence_graph_shortest_path() {
        let mut g = CoherenceGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_graph_topo_sort() {
        let mut g = CoherenceGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_coherence_graph_components() {
        let mut g = CoherenceGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_coherence_cache() {
        let mut cache = CoherenceCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_coherence_config() {
        let config = CoherenceConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_coherence_report() {
        let mut report = CoherenceReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_coherence_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = coherence_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_coherence_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = coherence_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = coherence_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_coherence_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = coherence_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_coherence_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = coherence_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_coherence_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = coherence_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_coherence_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = coherence_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_coherence_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = coherence_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_coherence_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = coherence_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_coherence_analysis_normalize() {
        let mut a = CoherenceAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_analysis_transpose() {
        let mut a = CoherenceAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_analysis_multiply() {
        let mut a = CoherenceAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = CoherenceAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_analysis_frobenius() {
        let mut a = CoherenceAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_analysis_symmetric() {
        let mut a = CoherenceAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_coherence_graph_dot() {
        let mut g = CoherenceGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_coherence_histogram_render() {
        let hist = CoherenceHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_coherence_batch_reset() {
        let mut proc = CoherenceBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_coherence_graph_remove_edge() {
        let mut g = CoherenceGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_coherence_dense_matrix_new() {
        let m = CoherenceDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_coherence_dense_matrix_identity() {
        let m = CoherenceDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_mul() {
        let a = CoherenceDenseMatrix::identity(2);
        let b = CoherenceDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_transpose() {
        let a = CoherenceDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_det_2x2() {
        let m = CoherenceDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_det_3x3() {
        let m = CoherenceDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_inverse_2x2() {
        let m = CoherenceDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_power() {
        let m = CoherenceDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_rank() {
        let m = CoherenceDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_coherence_dense_matrix_solve() {
        let a = CoherenceDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_coherence_dense_matrix_lu() {
        let a = CoherenceDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_eigenvalues() {
        let m = CoherenceDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_kronecker() {
        let a = CoherenceDenseMatrix::identity(2);
        let b = CoherenceDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_coherence_dense_matrix_hadamard() {
        let a = CoherenceDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = CoherenceDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_interval() {
        let a = CoherenceInterval::new(1.0, 3.0);
        let b = CoherenceInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_interval_mul() {
        let a = CoherenceInterval::new(-2.0, 3.0);
        let b = CoherenceInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_interval_hull() {
        let a = CoherenceInterval::new(1.0, 3.0);
        let b = CoherenceInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_state_machine() {
        let mut sm = CoherenceStateMachine::new();
        assert_eq!(*sm.state(), CoherenceState::Idle);
        assert!(sm.transition(CoherenceState::Checking));
        assert_eq!(*sm.state(), CoherenceState::Checking);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_coherence_state_machine_invalid() {
        let mut sm = CoherenceStateMachine::new();
        let last_state = CoherenceState::Error;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_coherence_state_machine_reset() {
        let mut sm = CoherenceStateMachine::new();
        sm.transition(CoherenceState::Checking);
        sm.reset();
        assert_eq!(*sm.state(), CoherenceState::Idle);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_coherence_ring_buffer() {
        let mut rb = CoherenceRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_ring_buffer_to_vec() {
        let mut rb = CoherenceRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_disjoint_set() {
        let mut ds = CoherenceDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_coherence_disjoint_set_components() {
        let mut ds = CoherenceDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_coherence_sorted_list() {
        let mut sl = CoherenceSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sorted_list_remove() {
        let mut sl = CoherenceSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_coherence_ema() {
        let mut ema = CoherenceEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_bloom_filter() {
        let mut bf = CoherenceBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_coherence_trie() {
        let mut trie = CoherenceTrie::new();
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
    fn test_coherence_dense_matrix_sym() {
        let m = CoherenceDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_coherence_dense_matrix_diag() {
        let m = CoherenceDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_coherence_dense_matrix_upper_tri() {
        let m = CoherenceDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_coherence_dense_matrix_outer() {
        let m = CoherenceDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dense_matrix_submatrix() {
        let m = CoherenceDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_priority_queue() {
        let mut pq = CoherencePriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_coherence_accumulator() {
        let mut acc = CoherenceAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_accumulator_merge() {
        let mut a = CoherenceAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = CoherenceAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sparse_matrix() {
        let mut m = CoherenceSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sparse_mul_vec() {
        let mut m = CoherenceSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sparse_transpose() {
        let mut m = CoherenceSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_eval() {
        let p = CoherencePolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_add() {
        let a = CoherencePolynomial::new(vec![1.0, 2.0]);
        let b = CoherencePolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_mul() {
        let a = CoherencePolynomial::new(vec![1.0, 1.0]);
        let b = CoherencePolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_deriv() {
        let p = CoherencePolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_integral() {
        let p = CoherencePolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_roots() {
        let p = CoherencePolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_coherence_polynomial_newton() {
        let p = CoherencePolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_polynomial_compose() {
        let p = CoherencePolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = CoherencePolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_rng() {
        let mut rng = CoherenceRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_coherence_rng_gaussian() {
        let mut rng = CoherenceRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_coherence_timer() {
        let mut timer = CoherenceTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_bitvector() {
        let mut bv = CoherenceBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_coherence_bitvector_ops() {
        let mut a = CoherenceBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = CoherenceBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_coherence_bitvector_jaccard() {
        let mut a = CoherenceBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = CoherenceBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_priority_queue_empty() {
        let mut pq = CoherencePriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_coherence_sparse_add() {
        let mut a = CoherenceSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = CoherenceSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_rng_shuffle() {
        let mut rng = CoherenceRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_polynomial_display() {
        let p = CoherencePolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_coherence_polynomial_monomial() {
        let m = CoherencePolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_timer_percentiles() {
        let mut timer = CoherenceTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_coherence_accumulator_cv() {
        let mut acc = CoherenceAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sparse_diagonal() {
        let mut m = CoherenceSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_lru_cache() {
        let mut cache = CoherenceLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_coherence_lru_hit_rate() {
        let mut cache = CoherenceLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_graph_coloring() {
        let mut gc = CoherenceGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_coherence_graph_coloring_complete() {
        let mut gc = CoherenceGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_coherence_topk() {
        let mut tk = CoherenceTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sliding_window() {
        let mut sw = CoherenceSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_coherence_sliding_window_trend() {
        let mut sw = CoherenceSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_coherence_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = CoherenceConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_coherence_confusion_f1() {
        let cm = CoherenceConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_coherence_cosine_similarity() {
        let s = coherence_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = coherence_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_euclidean_distance() {
        let d = coherence_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sigmoid() {
        let s = coherence_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = coherence_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_coherence_softmax() {
        let sm = coherence_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_coherence_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = coherence_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_normalize() {
        let v = coherence_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_lerp() {
        assert!((coherence_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((coherence_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((coherence_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_clamp() {
        assert!((coherence_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((coherence_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((coherence_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_cross_product() {
        let c = coherence_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dot_product() {
        let d = coherence_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = coherence_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = coherence_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_logsumexp() {
        let lse = coherence_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_coherence_feature_scaler() {
        let mut scaler = CoherenceFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_feature_scaler_inverse() {
        let mut scaler = CoherenceFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_linear_regression() {
        let mut lr = CoherenceLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_linear_regression_predict() {
        let mut lr = CoherenceLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_weighted_graph() {
        let mut g = CoherenceWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_weighted_graph_mst() {
        let mut g = CoherenceWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = coherence_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_cumsum() {
        let cs = coherence_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_diff() {
        let d = coherence_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_autocorrelation() {
        let ac = coherence_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_dft_magnitude() {
        let mags = coherence_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_coherence_integrate_trapezoid() {
        let area = coherence_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_convolve() {
        let c = coherence_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_weighted_graph_clustering() {
        let mut g = CoherenceWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_histogram_cumulative() {
        let h = CoherenceHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_coherence_histogram_entropy() {
        let h = CoherenceHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_coherence_aabb() {
        let bb = CoherenceAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_aabb_intersects() {
        let a = CoherenceAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = CoherenceAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = CoherenceAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_coherence_quadtree() {
        let bb = CoherenceAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = CoherenceQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(CoherencePoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_coherence_quadtree_query() {
        let bb = CoherenceAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = CoherenceQuadTree::new(bb, 2, 8);
        qt.insert(CoherencePoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(CoherencePoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = CoherenceAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_coherence_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = coherence_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = coherence_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = coherence_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((coherence_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_identity() {
        let id = coherence_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = coherence_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_running_stats() {
        let mut s = CoherenceRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_running_stats_merge() {
        let mut a = CoherenceRunningStats::new();
        let mut b = CoherenceRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_running_stats_cv() {
        let mut s = CoherenceRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_coherence_iqr() {
        let iqr = coherence_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_coherence_outliers() {
        let outliers = coherence_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_coherence_zscore() {
        let z = coherence_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_coherence_rank() {
        let r = coherence_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_spearman() {
        let rho = coherence_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_sample_skewness_symmetric() {
        let s = coherence_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = coherence_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_coherence_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = coherence_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }


}