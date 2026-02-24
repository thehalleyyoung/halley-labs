//! Execution graph representation with bit-packed relation storage.
//!
//! Core data structures for representing concurrent program executions
//! as graphs with events and relations (program order, reads-from,
//! coherence, from-reads). Uses `bitvec` for SIMD-friendly bit-parallel
//! operations on relation matrices.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use bitvec::prelude::*;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Fundamental type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for an event within an execution.
pub type EventId = usize;

/// Thread (or hardware context / CTA) identifier.
pub type ThreadId = usize;

/// Memory address (abstract).
pub type Address = u64;

/// Data value.
pub type Value = u64;

// ---------------------------------------------------------------------------
// OpType / Scope
// ---------------------------------------------------------------------------

/// The kind of memory operation an event represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    Read,
    Write,
    Fence,
    /// Read-Modify-Write (e.g., CAS, fetch-add).
    RMW,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Read  => write!(f, "R"),
            OpType::Write => write!(f, "W"),
            OpType::Fence => write!(f, "F"),
            OpType::RMW   => write!(f, "RMW"),
        }
    }
}

/// Scope annotation for GPU memory models (PTX / WebGPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// Cooperative Thread Array (warp-level in PTX).
    CTA,
    /// GPU device scope.
    GPU,
    /// System-wide scope.
    System,
    /// No explicit scope (CPU models).
    None,
}

impl Default for Scope {
    fn default() -> Self { Scope::None }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scope::CTA    => write!(f, ".cta"),
            Scope::GPU    => write!(f, ".gpu"),
            Scope::System => write!(f, ".sys"),
            Scope::None   => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A single event in an execution graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub thread: ThreadId,
    pub op_type: OpType,
    pub address: Address,
    pub value: Value,
    pub scope: Scope,
    /// Per-thread sequence number (program order index).
    pub po_index: usize,
}

impl Event {
    pub fn new(
        id: EventId,
        thread: ThreadId,
        op_type: OpType,
        address: Address,
        value: Value,
    ) -> Self {
        Self { id, thread, op_type, address, value, scope: Scope::None, po_index: 0 }
    }

    pub fn with_scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    pub fn with_po_index(mut self, idx: usize) -> Self {
        self.po_index = idx;
        self
    }

    pub fn is_read(&self) -> bool  { matches!(self.op_type, OpType::Read | OpType::RMW) }
    pub fn is_write(&self) -> bool { matches!(self.op_type, OpType::Write | OpType::RMW) }
    pub fn is_fence(&self) -> bool { matches!(self.op_type, OpType::Fence) }
    pub fn is_rmw(&self) -> bool   { matches!(self.op_type, OpType::RMW) }

    /// Short label for DOT output.
    pub fn label(&self) -> String {
        match self.op_type {
            OpType::Fence => format!("F{}", self.scope),
            _ => format!("{}{}({:#x})={}", self.op_type, self.scope, self.address, self.value),
        }
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}:T{}:{}", self.id, self.thread, self.label())
    }
}

// ---------------------------------------------------------------------------
// BitMatrix — bit-packed binary relation
// ---------------------------------------------------------------------------

/// A square binary relation over `n` elements stored as a flat bit-vector.
///
/// Entry (i, j) is at bit index `i * n + j`. This layout is SIMD-friendly
/// and allows full-row operations (union, intersection) to proceed in
/// O(n/64) steps.
#[derive(Clone, Serialize, Deserialize)]
pub struct BitMatrix {
    n: usize,
    bits: BitVec<u64, Lsb0>,
}

impl fmt::Debug for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitMatrix({}x{}, {} edges)", self.n, self.n, self.count_edges())
    }
}

impl PartialEq for BitMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.bits == other.bits
    }
}
impl Eq for BitMatrix {}

impl Hash for BitMatrix {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n.hash(state);
        // Hash underlying words for speed.
        for word in self.bits.as_raw_slice() {
            word.hash(state);
        }
    }
}

impl BitMatrix {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create an `n × n` zero matrix (empty relation).
    pub fn new(n: usize) -> Self {
        Self {
            n,
            bits: bitvec![u64, Lsb0; 0; n * n],
        }
    }

    /// Create the identity relation (diagonal).
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// Create a universal relation (all pairs).
    pub fn universal(n: usize) -> Self {
        Self {
            n,
            bits: bitvec![u64, Lsb0; 1; n * n],
        }
    }

    /// Dimension of the matrix.
    pub fn dim(&self) -> usize { self.n }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.n && j < self.n);
        i * self.n + j
    }

    /// Test whether (i, j) is in the relation.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.bits[self.idx(i, j)]
    }

    /// Set (i, j) in the relation.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        let idx = self.idx(i, j);
        self.bits.set(idx, val);
    }

    /// Add edge (i, j). Returns whether it was newly inserted.
    #[inline]
    pub fn add(&mut self, i: usize, j: usize) -> bool {
        let idx = self.idx(i, j);
        let prev = self.bits[idx];
        self.bits.set(idx, true);
        !prev
    }

    /// Remove edge (i, j).
    #[inline]
    pub fn remove(&mut self, i: usize, j: usize) {
        let idx = self.idx(i, j);
        self.bits.set(idx, false);
    }

    // -----------------------------------------------------------------------
    // Row / column operations
    // -----------------------------------------------------------------------

    /// Slice for row `i` (bits [i*n .. (i+1)*n]).
    fn row_range(&self, i: usize) -> std::ops::Range<usize> {
        let start = i * self.n;
        start..start + self.n
    }

    /// Iterate over successors of `i` (j such that (i,j) ∈ R).
    pub fn successors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let range = self.row_range(i);
        self.bits[range].iter_ones()
    }

    /// Iterate over predecessors of `j` (i such that (i,j) ∈ R).
    pub fn predecessors(&self, j: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(move |&i| self.get(i, j))
    }

    /// Number of edges in the relation.
    pub fn count_edges(&self) -> usize {
        self.bits.count_ones()
    }

    /// Whether the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.not_any()
    }

    /// Iterate all edges (i, j).
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for i in 0..self.n {
            for j in self.successors(i) {
                result.push((i, j));
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Set-theoretic operations (bit-parallel)
    // -----------------------------------------------------------------------

    /// Union: R1 ∪ R2.
    pub fn union(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = self.clone();
        result.bits |= &other.bits;
        result
    }

    /// In-place union.
    pub fn union_assign(&mut self, other: &Self) {
        assert_eq!(self.n, other.n);
        self.bits |= &other.bits;
    }

    /// Intersection: R1 ∩ R2.
    pub fn intersection(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = self.clone();
        result.bits &= &other.bits;
        result
    }

    /// In-place intersection.
    pub fn intersect_assign(&mut self, other: &Self) {
        assert_eq!(self.n, other.n);
        self.bits &= &other.bits;
    }

    /// Complement: ¬R (with respect to n×n universal).
    pub fn complement(&self) -> Self {
        let mut result = self.clone();
        for i in 0..result.bits.len() {
            let v = result.bits[i];
            result.bits.set(i, !v);
        }
        result
    }

    /// Difference: R1 \ R2.
    pub fn difference(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let comp = other.complement();
        self.intersection(&comp)
    }

    // -----------------------------------------------------------------------
    // Relational algebra (bit-parallel)
    // -----------------------------------------------------------------------

    /// Inverse (transpose): R^{-1}.
    pub fn inverse(&self) -> Self {
        let mut result = Self::new(self.n);
        for i in 0..self.n {
            for j in self.successors(i) {
                result.set(j, i, true);
            }
        }
        result
    }

    /// Sequence (composition): R1 ; R2 = { (a,c) | ∃b. (a,b)∈R1 ∧ (b,c)∈R2 }.
    ///
    /// Implemented as bit-parallel matrix multiply over GF(2) (OR instead of XOR).
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let mut result = Self::new(n);
        // Transpose `other` so we can do row-row dot products.
        let other_t = other.inverse();
        for i in 0..n {
            let row_a_start = i * n;
            for j in 0..n {
                let row_bt_start = j * n;
                // Check if row_a AND row_bt is non-zero.
                let mut found = false;
                let a_slice = &self.bits[row_a_start..row_a_start + n];
                let b_slice = &other_t.bits[row_bt_start..row_bt_start + n];
                // Bit-parallel AND + any
                for k in 0..a_slice.len() {
                    if a_slice[k] && b_slice[k] {
                        found = true;
                        break;
                    }
                }
                if found {
                    result.set(i, j, true);
                }
            }
        }
        result
    }

    /// Transitive closure via Warshall's algorithm (bit-parallel).
    ///
    /// Complexity: O(n³ / 64) where 64 is the word size.
    pub fn transitive_closure(&self) -> Self {
        let n = self.n;
        let mut tc = self.clone();
        for k in 0..n {
            for i in 0..n {
                if tc.get(i, k) {
                    // tc[i] |= tc[k]
                    let k_start = k * n;
                    let i_start = i * n;
                    let row_k: BitVec<u64, Lsb0> = tc.bits[k_start..k_start + n].to_bitvec();
                    for idx in 0..n {
                        if row_k[idx] {
                            tc.bits.set(i_start + idx, true);
                        }
                    }
                }
            }
        }
        tc
    }

    /// Reflexive-transitive closure: R* = Id ∪ R⁺.
    pub fn reflexive_transitive_closure(&self) -> Self {
        self.transitive_closure().union(&Self::identity(self.n))
    }

    /// Transitive closure (alias for clarity in model definitions).
    pub fn plus(&self) -> Self {
        self.transitive_closure()
    }

    /// R? = Id ∪ R (optional / reflexive closure).
    pub fn optional(&self) -> Self {
        self.union(&Self::identity(self.n))
    }

    /// Identity restriction: [P] maps event set P to the identity on P.
    /// `predicate` is a boolean vector of length n.
    pub fn identity_filter(n: usize, predicate: &[bool]) -> Self {
        assert_eq!(predicate.len(), n);
        let mut m = Self::new(n);
        for (i, &keep) in predicate.iter().enumerate() {
            if keep {
                m.set(i, i, true);
            }
        }
        m
    }

    /// Restrict relation to pairs where source satisfies `pred_src` and
    /// target satisfies `pred_dst`.
    pub fn restrict(&self, pred_src: &[bool], pred_dst: &[bool]) -> Self {
        let n = self.n;
        assert_eq!(pred_src.len(), n);
        assert_eq!(pred_dst.len(), n);
        let mut result = self.clone();
        for i in 0..n {
            if !pred_src[i] {
                for j in 0..n {
                    result.set(i, j, false);
                }
            } else {
                for j in 0..n {
                    if !pred_dst[j] {
                        result.set(i, j, false);
                    }
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Cycle detection
    // -----------------------------------------------------------------------

    /// Check whether the relation is acyclic (no cycles in the directed graph).
    ///
    /// Uses Kahn's algorithm (topological sort) for O(n + E) detection.
    pub fn is_acyclic(&self) -> bool {
        let n = self.n;
        let mut in_degree = vec![0u32; n];
        let mut has_edges = vec![false; n];

        for i in 0..n {
            for j in self.successors(i) {
                if i != j {
                    in_degree[j] += 1;
                    has_edges[i] = true;
                    has_edges[j] = true;
                }
            }
        }

        // Self-loops are instant cycles.
        for i in 0..n {
            if self.get(i, i) {
                return false;
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 && has_edges[i] {
                queue.push_back(i);
            }
        }

        let active_count = (0..n).filter(|&i| has_edges[i]).count();
        let mut removed = 0usize;

        while let Some(u) = queue.pop_front() {
            removed += 1;
            for v in self.successors(u) {
                if u == v { continue; }
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        removed >= active_count
    }

    /// Check whether the relation is irreflexive (no self-loops).
    pub fn is_irreflexive(&self) -> bool {
        for i in 0..self.n {
            if self.get(i, i) {
                return false;
            }
        }
        true
    }

    /// Find a cycle if one exists. Returns `None` if acyclic.
    pub fn find_cycle(&self) -> Option<Vec<usize>> {
        let n = self.n;
        let mut color = vec![0u8; n]; // 0=white, 1=gray, 2=black
        let mut parent = vec![usize::MAX; n];

        for start in 0..n {
            if color[start] != 0 { continue; }
            let mut stack = vec![(start, false)];
            while let Some((u, processed)) = stack.pop() {
                if processed {
                    color[u] = 2;
                    continue;
                }
                if color[u] == 1 {
                    color[u] = 2;
                    continue;
                }
                color[u] = 1;
                stack.push((u, true));
                for v in self.successors(u) {
                    if color[v] == 1 {
                        // Found cycle: reconstruct.
                        let mut cycle = vec![v, u];
                        let mut cur = u;
                        while cur != v && parent[cur] != usize::MAX {
                            cur = parent[cur];
                            if cur == v { break; }
                            cycle.push(cur);
                        }
                        cycle.reverse();
                        return Some(cycle);
                    }
                    if color[v] == 0 {
                        parent[v] = u;
                        stack.push((v, false));
                    }
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Topological sort
    // -----------------------------------------------------------------------

    /// Topological sort. Returns `None` if the graph has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.n;
        let mut in_deg = vec![0u32; n];
        for i in 0..n {
            for j in self.successors(i) {
                if i != j { in_deg[j] += 1; }
            }
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for v in self.successors(u) {
                if u == v { continue; }
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if order.len() == n { Some(order) } else { None }
    }

    // -----------------------------------------------------------------------
    // Visualization helpers
    // -----------------------------------------------------------------------

    /// Pretty-print as adjacency matrix.
    pub fn pretty_print(&self) -> String {
        let n = self.n;
        let mut s = String::new();
        s.push_str(&format!("   "));
        for j in 0..n {
            s.push_str(&format!("{:>3}", j));
        }
        s.push('\n');
        for i in 0..n {
            s.push_str(&format!("{:>2} ", i));
            for j in 0..n {
                s.push_str(if self.get(i, j) { "  1" } else { "  ." });
            }
            s.push('\n');
        }
        s
    }

    /// Generate DOT edges for this relation, using the given label.
    pub fn dot_edges(&self, label: &str, color: &str) -> String {
        let mut s = String::new();
        for (i, j) in self.edges() {
            s.push_str(&format!(
                "  e{} -> e{} [label=\"{}\", color=\"{}\", fontcolor=\"{}\"];\n",
                i, j, label, color, color
            ));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Named relation wrapper
// ---------------------------------------------------------------------------

/// A named relation (for storage in execution graphs / models).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub name: String,
    pub matrix: BitMatrix,
}

impl Relation {
    pub fn new(name: impl Into<String>, matrix: BitMatrix) -> Self {
        Self { name: name.into(), matrix }
    }
}

// ---------------------------------------------------------------------------
// ExecutionGraph
// ---------------------------------------------------------------------------

/// Complete execution graph: events + base relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub events: Vec<Event>,
    /// Program order.
    pub po: BitMatrix,
    /// Reads-from.
    pub rf: BitMatrix,
    /// Coherence (total order per address on writes).
    pub co: BitMatrix,
    /// From-reads (derived: rf⁻¹ ; co).
    pub fr: BitMatrix,
    /// Optional additional named relations.
    pub extra: Vec<Relation>,
    /// Cache: event index by thread.
    #[serde(skip)]
    thread_events: HashMap<ThreadId, Vec<EventId>>,
    /// Cache: event index by address.
    #[serde(skip)]
    addr_events: HashMap<Address, Vec<EventId>>,
}

impl ExecutionGraph {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build an empty execution graph.
    pub fn empty() -> Self {
        Self {
            events: Vec::new(),
            po: BitMatrix::new(0),
            rf: BitMatrix::new(0),
            co: BitMatrix::new(0),
            fr: BitMatrix::new(0),
            extra: Vec::new(),
            thread_events: HashMap::new(),
            addr_events: HashMap::new(),
        }
    }

    /// Build an execution graph from events, computing `po` automatically.
    pub fn new(events: Vec<Event>) -> Self {
        let n = events.len();
        let mut po = BitMatrix::new(n);

        // Build thread_events map.
        let mut thread_events: HashMap<ThreadId, Vec<EventId>> = HashMap::new();
        let mut addr_events: HashMap<Address, Vec<EventId>> = HashMap::new();
        for e in &events {
            thread_events.entry(e.thread).or_default().push(e.id);
            if e.op_type != OpType::Fence {
                addr_events.entry(e.address).or_default().push(e.id);
            }
        }

        // Program order: within each thread, earlier po_index → later po_index.
        for (_tid, evts) in &thread_events {
            let mut sorted = evts.clone();
            sorted.sort_by_key(|&eid| events[eid].po_index);
            for i in 0..sorted.len() {
                for j in i + 1..sorted.len() {
                    po.set(sorted[i], sorted[j], true);
                }
            }
        }

        Self {
            events,
            po,
            rf: BitMatrix::new(n),
            co: BitMatrix::new(n),
            fr: BitMatrix::new(n),
            extra: Vec::new(),
            thread_events,
            addr_events,
        }
    }

    /// Number of events.
    pub fn len(&self) -> usize { self.events.len() }

    /// Whether the graph has no events.
    pub fn is_empty(&self) -> bool { self.events.is_empty() }

    /// Rebuild internal caches after deserialization.
    pub fn rebuild_caches(&mut self) {
        self.thread_events.clear();
        self.addr_events.clear();
        for e in &self.events {
            self.thread_events.entry(e.thread).or_default().push(e.id);
            if e.op_type != OpType::Fence {
                self.addr_events.entry(e.address).or_default().push(e.id);
            }
        }
    }

    /// Get events for a given thread.
    pub fn thread_events(&self, tid: ThreadId) -> &[EventId] {
        self.thread_events.get(&tid).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get events accessing a given address.
    pub fn addr_events(&self, addr: Address) -> &[EventId] {
        self.addr_events.get(&addr).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Distinct thread ids.
    pub fn thread_ids(&self) -> Vec<ThreadId> {
        let mut tids: Vec<ThreadId> = self.thread_events.keys().copied().collect();
        tids.sort();
        tids
    }

    /// Distinct addresses (excluding fence events).
    pub fn addresses(&self) -> Vec<Address> {
        let mut addrs: Vec<Address> = self.addr_events.keys().copied().collect();
        addrs.sort();
        addrs
    }

    // -----------------------------------------------------------------------
    // Relation manipulation
    // -----------------------------------------------------------------------

    /// Add a reads-from edge: write `w` is read by `r`.
    pub fn add_rf(&mut self, w: EventId, r: EventId) {
        assert!(self.events[w].is_write(), "rf source must be a write");
        assert!(self.events[r].is_read(), "rf target must be a read");
        assert_eq!(self.events[w].address, self.events[r].address,
                   "rf must be same-address");
        self.rf.set(w, r, true);
    }

    /// Add a coherence edge: write `w1` is co-before `w2`.
    pub fn add_co(&mut self, w1: EventId, w2: EventId) {
        assert!(self.events[w1].is_write(), "co source must be a write");
        assert!(self.events[w2].is_write(), "co target must be a write");
        assert_eq!(self.events[w1].address, self.events[w2].address,
                   "co must be same-address");
        self.co.set(w1, w2, true);
    }

    /// Derive the from-reads relation: fr = rf⁻¹ ; co.
    pub fn derive_fr(&mut self) {
        self.fr = self.rf.inverse().compose(&self.co);
    }

    /// Internal consistency of reads-from / coherence.
    /// Sets `co` to full per-address total order among writes and derives `fr`.
    pub fn set_co_total_order(&mut self, addr: Address, write_order: &[EventId]) {
        for i in 0..write_order.len() {
            for j in i + 1..write_order.len() {
                self.co.set(write_order[i], write_order[j], true);
            }
        }
    }

    /// Add an extra named relation.
    pub fn add_relation(&mut self, name: impl Into<String>, matrix: BitMatrix) {
        self.extra.push(Relation::new(name, matrix));
    }

    /// Get a named extra relation.
    pub fn get_relation(&self, name: &str) -> Option<&BitMatrix> {
        self.extra.iter().find(|r| r.name == name).map(|r| &r.matrix)
    }

    /// Build a predicate vector (boolean per event) for events matching a filter.
    pub fn predicate<F: Fn(&Event) -> bool>(&self, f: F) -> Vec<bool> {
        self.events.iter().map(|e| f(e)).collect()
    }

    /// Reads predicate.
    pub fn reads_pred(&self) -> Vec<bool> { self.predicate(|e| e.is_read()) }

    /// Writes predicate.
    pub fn writes_pred(&self) -> Vec<bool> { self.predicate(|e| e.is_write()) }

    /// Fence predicate.
    pub fn fences_pred(&self) -> Vec<bool> { self.predicate(|e| e.is_fence()) }

    /// RMW predicate.
    pub fn rmw_pred(&self) -> Vec<bool> { self.predicate(|e| e.is_rmw()) }

    /// Same-address predicate matrix.
    pub fn same_address(&self) -> BitMatrix {
        let n = self.len();
        let mut m = BitMatrix::new(n);
        for (&addr, evts) in &self.addr_events {
            let _ = addr;
            for &a in evts {
                for &b in evts {
                    m.set(a, b, true);
                }
            }
        }
        m
    }

    /// Same-thread predicate matrix.
    pub fn same_thread(&self) -> BitMatrix {
        let n = self.len();
        let mut m = BitMatrix::new(n);
        for (_tid, evts) in &self.thread_events {
            for &a in evts {
                for &b in evts {
                    m.set(a, b, true);
                }
            }
        }
        m
    }

    /// External relation: edges between different threads only.
    pub fn external(&self, rel: &BitMatrix) -> BitMatrix {
        let int = self.same_thread();
        rel.difference(&int)
    }

    /// Internal relation: edges within same thread only.
    pub fn internal(&self, rel: &BitMatrix) -> BitMatrix {
        rel.intersection(&self.same_thread())
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate the structural consistency of the execution graph.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let n = self.len();

        // Check dimensions.
        if self.po.dim() != n { errors.push(format!("po dimension mismatch: {} vs {}", self.po.dim(), n)); }
        if self.rf.dim() != n { errors.push(format!("rf dimension mismatch: {} vs {}", self.rf.dim(), n)); }
        if self.co.dim() != n { errors.push(format!("co dimension mismatch: {} vs {}", self.co.dim(), n)); }
        if self.fr.dim() != n { errors.push(format!("fr dimension mismatch: {} vs {}", self.fr.dim(), n)); }

        // Check event ids are sequential.
        for (i, e) in self.events.iter().enumerate() {
            if e.id != i {
                errors.push(format!("event {} has id {}", i, e.id));
            }
        }

        // rf: write → read, same address.
        for (w, r) in self.rf.edges() {
            if !self.events[w].is_write() {
                errors.push(format!("rf source e{} is not a write", w));
            }
            if !self.events[r].is_read() {
                errors.push(format!("rf target e{} is not a read", r));
            }
            if self.events[w].address != self.events[r].address {
                errors.push(format!("rf e{}→e{} different addresses", w, r));
            }
        }

        // co: write → write, same address.
        for (w1, w2) in self.co.edges() {
            if !self.events[w1].is_write() {
                errors.push(format!("co source e{} is not a write", w1));
            }
            if !self.events[w2].is_write() {
                errors.push(format!("co target e{} is not a write", w2));
            }
            if self.events[w1].address != self.events[w2].address {
                errors.push(format!("co e{}→e{} different addresses", w1, w2));
            }
        }

        // Each read should have at most one rf source.
        for r in 0..n {
            if self.events[r].is_read() {
                let sources: Vec<_> = self.rf.predecessors(r).collect();
                if sources.len() > 1 {
                    errors.push(format!("read e{} has {} rf sources", r, sources.len()));
                }
            }
        }

        // co should be a strict partial order (irreflexive, acyclic) per address.
        if !self.co.is_irreflexive() {
            errors.push("co has self-loops".into());
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    // -----------------------------------------------------------------------
    // Comparison / hashing
    // -----------------------------------------------------------------------

    /// Structural equality (events + all relations).
    pub fn structurally_equal(&self, other: &Self) -> bool {
        self.events == other.events
            && self.po == other.po
            && self.rf == other.rf
            && self.co == other.co
            && self.fr == other.fr
    }

    /// Compute a hash suitable for deduplication.
    pub fn structural_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.events.hash(&mut hasher);
        self.po.hash(&mut hasher);
        self.rf.hash(&mut hasher);
        self.co.hash(&mut hasher);
        self.fr.hash(&mut hasher);
        hasher.finish()
    }

    // -----------------------------------------------------------------------
    // Pretty printing
    // -----------------------------------------------------------------------

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "ExecutionGraph: {} events, {} threads, {} addrs, po={} rf={} co={} fr={}",
            self.len(),
            self.thread_ids().len(),
            self.addresses().len(),
            self.po.count_edges(),
            self.rf.count_edges(),
            self.co.count_edges(),
            self.fr.count_edges(),
        )
    }

    /// Generate Graphviz DOT representation.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph execution {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=record, fontsize=10];\n\n");

        // Group events by thread.
        for tid in self.thread_ids() {
            dot.push_str(&format!("  subgraph cluster_t{} {{\n", tid));
            dot.push_str(&format!("    label=\"Thread {}\";\n", tid));
            dot.push_str("    style=dashed;\n");
            for &eid in self.thread_events(tid) {
                let e = &self.events[eid];
                dot.push_str(&format!("    e{} [label=\"{}\"];\n", eid, e.label()));
            }
            dot.push_str("  }\n\n");
        }

        // Relations.
        dot.push_str(&self.po.dot_edges("po", "black"));
        dot.push_str(&self.rf.dot_edges("rf", "red"));
        dot.push_str(&self.co.dot_edges("co", "blue"));
        dot.push_str(&self.fr.dot_edges("fr", "green"));
        for rel in &self.extra {
            dot.push_str(&rel.matrix.dot_edges(&rel.name, "purple"));
        }

        dot.push_str("}\n");
        dot
    }
}

impl fmt::Display for ExecutionGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary())?;
        writeln!(f, "Events:")?;
        for e in &self.events {
            writeln!(f, "  {}", e)?;
        }
        if !self.rf.is_empty() {
            writeln!(f, "rf: {:?}", self.rf.edges())?;
        }
        if !self.co.is_empty() {
            writeln!(f, "co: {:?}", self.co.edges())?;
        }
        if !self.fr.is_empty() {
            writeln!(f, "fr: {:?}", self.fr.edges())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ExecutionGraphBuilder — fluent construction
// ---------------------------------------------------------------------------

/// Fluent builder for constructing execution graphs.
pub struct ExecutionGraphBuilder {
    events: Vec<Event>,
    next_id: EventId,
    thread_counters: HashMap<ThreadId, usize>,
}

impl ExecutionGraphBuilder {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 0,
            thread_counters: HashMap::new(),
        }
    }

    fn next_po(&mut self, tid: ThreadId) -> usize {
        let c = self.thread_counters.entry(tid).or_insert(0);
        let v = *c;
        *c += 1;
        v
    }

    pub fn add_event(&mut self, thread: ThreadId, op: OpType, addr: Address, val: Value) -> EventId {
        let po = self.next_po(thread);
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(Event {
            id,
            thread,
            op_type: op,
            address: addr,
            value: val,
            scope: Scope::None,
            po_index: po,
        });
        id
    }

    pub fn add_read(&mut self, thread: ThreadId, addr: Address, val: Value) -> EventId {
        self.add_event(thread, OpType::Read, addr, val)
    }

    pub fn add_write(&mut self, thread: ThreadId, addr: Address, val: Value) -> EventId {
        self.add_event(thread, OpType::Write, addr, val)
    }

    pub fn add_fence(&mut self, thread: ThreadId) -> EventId {
        self.add_event(thread, OpType::Fence, 0, 0)
    }

    pub fn add_rmw(&mut self, thread: ThreadId, addr: Address, val: Value) -> EventId {
        self.add_event(thread, OpType::RMW, addr, val)
    }

    pub fn add_event_scoped(
        &mut self, thread: ThreadId, op: OpType, addr: Address, val: Value, scope: Scope,
    ) -> EventId {
        let id = self.add_event(thread, op, addr, val);
        self.events[id].scope = scope;
        id
    }

    pub fn build(self) -> ExecutionGraph {
        ExecutionGraph::new(self.events)
    }
}

impl Default for ExecutionGraphBuilder {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> ExecutionGraph {
        // T0: W(x)=1, R(x)=1
        // T1: W(x)=2
        let mut b = ExecutionGraphBuilder::new();
        let w0 = b.add_write(0, 0x100, 1);
        let r0 = b.add_read(0, 0x100, 1);
        let w1 = b.add_write(1, 0x100, 2);
        let mut g = b.build();
        g.add_rf(w0, r0);
        g.add_co(w0, w1);
        g.derive_fr();
        g
    }

    #[test]
    fn test_bitmatrix_basics() {
        let mut m = BitMatrix::new(4);
        assert!(m.is_empty());
        m.set(0, 1, true);
        m.set(1, 2, true);
        assert_eq!(m.count_edges(), 2);
        assert!(m.get(0, 1));
        assert!(!m.get(1, 0));
    }

    #[test]
    fn test_bitmatrix_identity() {
        let id = BitMatrix::identity(3);
        assert!(id.get(0, 0));
        assert!(id.get(1, 1));
        assert!(id.get(2, 2));
        assert!(!id.get(0, 1));
        assert_eq!(id.count_edges(), 3);
    }

    #[test]
    fn test_bitmatrix_union_intersection() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        a.set(1, 2, true);

        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        b.set(2, 0, true);

        let u = a.union(&b);
        assert_eq!(u.count_edges(), 3);

        let i = a.intersection(&b);
        assert_eq!(i.count_edges(), 1);
        assert!(i.get(1, 2));
    }

    #[test]
    fn test_bitmatrix_inverse() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        let inv = m.inverse();
        assert!(inv.get(1, 0));
        assert!(inv.get(2, 0));
        assert!(!inv.get(0, 1));
    }

    #[test]
    fn test_bitmatrix_compose() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        let c = a.compose(&b);
        assert!(c.get(0, 2));
        assert!(!c.get(0, 1));
        assert_eq!(c.count_edges(), 1);
    }

    #[test]
    fn test_bitmatrix_transitive_closure() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 3, true);
        let tc = m.transitive_closure();
        assert!(tc.get(0, 3));
        assert!(tc.get(0, 2));
        assert!(tc.get(1, 3));
        assert_eq!(tc.count_edges(), 6); // 0→1,0→2,0→3,1→2,1→3,2→3
    }

    #[test]
    fn test_bitmatrix_acyclic() {
        let mut dag = BitMatrix::new(3);
        dag.set(0, 1, true);
        dag.set(1, 2, true);
        assert!(dag.is_acyclic());

        let mut cyclic = BitMatrix::new(3);
        cyclic.set(0, 1, true);
        cyclic.set(1, 2, true);
        cyclic.set(2, 0, true);
        assert!(!cyclic.is_acyclic());
    }

    #[test]
    fn test_bitmatrix_find_cycle() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        let cycle = m.find_cycle();
        assert!(cycle.is_some());
        let c = cycle.unwrap();
        assert!(c.len() >= 2);
    }

    #[test]
    fn test_bitmatrix_topological_sort() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 3, true);
        m.set(2, 3, true);
        let order = m.topological_sort().unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(*order.last().unwrap(), 3);
    }

    #[test]
    fn test_bitmatrix_reflexive_transitive_closure() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        let rtc = m.reflexive_transitive_closure();
        // Identity entries.
        assert!(rtc.get(0, 0));
        assert!(rtc.get(1, 1));
        assert!(rtc.get(2, 2));
        // Transitive entries.
        assert!(rtc.get(0, 2));
    }

    #[test]
    fn test_bitmatrix_identity_filter() {
        let pred = vec![true, false, true, false];
        let m = BitMatrix::identity_filter(4, &pred);
        assert!(m.get(0, 0));
        assert!(!m.get(1, 1));
        assert!(m.get(2, 2));
        assert_eq!(m.count_edges(), 2);
    }

    #[test]
    fn test_bitmatrix_restrict() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let src = vec![true, false, true];
        let dst = vec![false, true, true];
        let r = m.restrict(&src, &dst);
        assert!(r.get(0, 1));
        assert!(r.get(0, 2));
        assert!(!r.get(1, 2)); // source filtered out
        assert_eq!(r.count_edges(), 2);
    }

    #[test]
    fn test_bitmatrix_difference() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        a.set(1, 2, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        let d = a.difference(&b);
        assert!(d.get(0, 1));
        assert!(!d.get(1, 2));
    }

    #[test]
    fn test_bitmatrix_optional() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        let opt = m.optional();
        assert!(opt.get(0, 0));
        assert!(opt.get(1, 1));
        assert!(opt.get(2, 2));
        assert!(opt.get(0, 1));
    }

    #[test]
    fn test_bitmatrix_pretty_print() {
        let mut m = BitMatrix::new(2);
        m.set(0, 1, true);
        let s = m.pretty_print();
        assert!(s.contains("1"));
    }

    #[test]
    fn test_bitmatrix_successors_predecessors() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(0, 3, true);
        m.set(2, 3, true);
        let succ: Vec<_> = m.successors(0).collect();
        assert_eq!(succ, vec![1, 3]);
        let pred: Vec<_> = m.predecessors(3).collect();
        assert_eq!(pred, vec![0, 2]);
    }

    #[test]
    fn test_execution_graph_construction() {
        let g = make_simple_graph();
        assert_eq!(g.len(), 3);
        assert_eq!(g.thread_ids(), vec![0, 1]);
        assert_eq!(g.addresses(), vec![0x100]);
        assert!(g.po.get(0, 1)); // W→R in thread 0
    }

    #[test]
    fn test_execution_graph_rf_co_fr() {
        let g = make_simple_graph();
        assert!(g.rf.get(0, 1)); // W(x)=1 → R(x)=1
        assert!(g.co.get(0, 2)); // W(x)=1 co W(x)=2
        // fr: R(x)=1 reads from W(x)=1 which is co-before W(x)=2
        // so R(x)=1 fr W(x)=2
        assert!(g.fr.get(1, 2));
    }

    #[test]
    fn test_execution_graph_validate() {
        let g = make_simple_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_execution_graph_predicates() {
        let g = make_simple_graph();
        let reads = g.reads_pred();
        assert_eq!(reads, vec![false, true, false]);
        let writes = g.writes_pred();
        assert_eq!(writes, vec![true, false, true]);
    }

    #[test]
    fn test_execution_graph_same_address() {
        let g = make_simple_graph();
        let sa = g.same_address();
        assert!(sa.get(0, 1));
        assert!(sa.get(0, 2));
        assert!(sa.get(1, 2));
    }

    #[test]
    fn test_execution_graph_same_thread() {
        let g = make_simple_graph();
        let st = g.same_thread();
        assert!(st.get(0, 1)); // both thread 0
        assert!(!st.get(0, 2)); // different threads
    }

    #[test]
    fn test_execution_graph_external_internal() {
        let g = make_simple_graph();
        let rfe = g.external(&g.rf);
        assert!(!rfe.get(0, 1)); // internal rf
        let rfi = g.internal(&g.rf);
        assert!(rfi.get(0, 1));
    }

    #[test]
    fn test_execution_graph_dot() {
        let g = make_simple_graph();
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("Thread 0"));
        assert!(dot.contains("Thread 1"));
    }

    #[test]
    fn test_execution_graph_summary() {
        let g = make_simple_graph();
        let s = g.summary();
        assert!(s.contains("3 events"));
    }

    #[test]
    fn test_execution_graph_structural_hash() {
        let g1 = make_simple_graph();
        let g2 = make_simple_graph();
        assert_eq!(g1.structural_hash(), g2.structural_hash());
    }

    #[test]
    fn test_builder_scoped_events() {
        let mut b = ExecutionGraphBuilder::new();
        let _w = b.add_event_scoped(0, OpType::Write, 0x100, 1, Scope::CTA);
        let g = b.build();
        assert_eq!(g.events[0].scope, Scope::CTA);
    }

    #[test]
    fn test_builder_fence() {
        let mut b = ExecutionGraphBuilder::new();
        let _w = b.add_write(0, 0x100, 1);
        let _f = b.add_fence(0);
        let _r = b.add_read(0, 0x100, 1);
        let g = b.build();
        assert!(g.po.get(0, 1)); // W → F
        assert!(g.po.get(1, 2)); // F → R
        assert!(g.po.get(0, 2)); // W → R
    }

    #[test]
    fn test_builder_rmw() {
        let mut b = ExecutionGraphBuilder::new();
        let rmw = b.add_rmw(0, 0x100, 42);
        let g = b.build();
        assert!(g.events[rmw].is_read());
        assert!(g.events[rmw].is_write());
        assert!(g.events[rmw].is_rmw());
    }

    #[test]
    fn test_execution_graph_rebuild_caches() {
        let mut g = make_simple_graph();
        g.thread_events.clear();
        g.addr_events.clear();
        g.rebuild_caches();
        assert_eq!(g.thread_events(0).len(), 2);
        assert_eq!(g.thread_events(1).len(), 1);
    }

    #[test]
    fn test_bitmatrix_dot_edges() {
        let mut m = BitMatrix::new(2);
        m.set(0, 1, true);
        let dot = m.dot_edges("rf", "red");
        assert!(dot.contains("e0 -> e1"));
        assert!(dot.contains("rf"));
    }

    #[test]
    fn test_large_bitmatrix() {
        let n = 128;
        let mut m = BitMatrix::new(n);
        for i in 0..n - 1 {
            m.set(i, i + 1, true);
        }
        assert_eq!(m.count_edges(), n - 1);
        assert!(m.is_acyclic());
        let tc = m.transitive_closure();
        assert_eq!(tc.count_edges(), n * (n - 1) / 2);
    }

    #[test]
    fn test_execution_graph_display() {
        let g = make_simple_graph();
        let s = format!("{}", g);
        assert!(s.contains("events"));
    }

    #[test]
    fn test_event_display() {
        let e = Event::new(0, 0, OpType::Write, 0x100, 42);
        let s = format!("{}", e);
        assert!(s.contains("W"));
        assert!(s.contains("0x100"));
    }
}
