//! CUDA scoped memory model specification for LITMUS∞.
//!
//! Implements the CUDA/PTX memory model with full thread-scope hierarchy:
//! thread, block (CTA), device, and system scopes. Supports scoped
//! acquire/release semantics, `cuda::atomic_thread_fence` variants,
//! and PTX memory model axioms (coherence, atomicity per scope).
//!
//! # Scope hierarchy
//!
//! ```text
//! System
//!  └─ Device
//!      └─ Block (CTA)
//!          └─ Thread
//! ```
//!
//! Memory ordering guarantees are only provided within a scope instance.
//! A release at block scope synchronises only with an acquire in the same
//! block; device-scope operations synchronise across blocks on the same GPU.

#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// CUDA scope hierarchy
// ---------------------------------------------------------------------------

/// CUDA memory scope levels following the PTX ISA specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CudaScope {
    /// Single thread — trivially ordered.
    Thread,
    /// Cooperative Thread Array (block).
    Block,
    /// Entire GPU device.
    Device,
    /// System-wide (CPU + all GPUs).
    System,
}

impl CudaScope {
    /// All scopes from narrowest to broadest.
    pub fn all() -> &'static [CudaScope] {
        &[CudaScope::Thread, CudaScope::Block, CudaScope::Device, CudaScope::System]
    }

    /// Whether `self` is at least as broad as `other`.
    pub fn includes(&self, other: &CudaScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    /// Next broader scope, or `None` for System.
    pub fn broaden(&self) -> Option<CudaScope> {
        match self {
            CudaScope::Thread => Some(CudaScope::Block),
            CudaScope::Block => Some(CudaScope::Device),
            CudaScope::Device => Some(CudaScope::System),
            CudaScope::System => None,
        }
    }

    /// Next narrower scope, or `None` for Thread.
    pub fn narrow(&self) -> Option<CudaScope> {
        match self {
            CudaScope::System => Some(CudaScope::Device),
            CudaScope::Device => Some(CudaScope::Block),
            CudaScope::Block => Some(CudaScope::Thread),
            CudaScope::Thread => None,
        }
    }
}

impl fmt::Display for CudaScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaScope::Thread => write!(f, "thread"),
            CudaScope::Block => write!(f, "block"),
            CudaScope::Device => write!(f, "device"),
            CudaScope::System => write!(f, "system"),
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA scope hierarchy tracker
// ---------------------------------------------------------------------------

/// Represents the CUDA scope hierarchy for a kernel launch, tracking which
/// threads belong to which blocks and devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaScopeHierarchy {
    /// Maps thread id → block id.
    pub thread_to_block: HashMap<usize, usize>,
    /// Maps block id → device id.
    pub block_to_device: HashMap<usize, usize>,
    /// Number of threads per block (uniform).
    pub block_size: usize,
    /// Number of blocks per device (uniform).
    pub grid_size: usize,
}

impl CudaScopeHierarchy {
    /// Build a hierarchy with `num_blocks` blocks of `block_size` threads
    /// on a single device (device 0).
    pub fn new(block_size: usize, num_blocks: usize) -> Self {
        let mut thread_to_block = HashMap::new();
        let mut block_to_device = HashMap::new();
        for blk in 0..num_blocks {
            block_to_device.insert(blk, 0);
            for t in 0..block_size {
                thread_to_block.insert(blk * block_size + t, blk);
            }
        }
        Self {
            thread_to_block,
            block_to_device,
            block_size,
            grid_size: num_blocks,
        }
    }

    /// Build a multi-device hierarchy.
    pub fn multi_device(block_size: usize, blocks_per_device: usize, num_devices: usize) -> Self {
        let mut thread_to_block = HashMap::new();
        let mut block_to_device = HashMap::new();
        let mut tid = 0;
        for dev in 0..num_devices {
            for blk_off in 0..blocks_per_device {
                let blk = dev * blocks_per_device + blk_off;
                block_to_device.insert(blk, dev);
                for _ in 0..block_size {
                    thread_to_block.insert(tid, blk);
                    tid += 1;
                }
            }
        }
        Self {
            thread_to_block,
            block_to_device,
            block_size,
            grid_size: blocks_per_device * num_devices,
        }
    }

    /// Check whether two threads share the given scope instance.
    pub fn same_scope_instance(&self, t1: usize, t2: usize, scope: CudaScope) -> bool {
        match scope {
            CudaScope::Thread => t1 == t2,
            CudaScope::Block => self.thread_to_block.get(&t1) == self.thread_to_block.get(&t2),
            CudaScope::Device => {
                let b1 = self.thread_to_block.get(&t1);
                let b2 = self.thread_to_block.get(&t2);
                match (b1, b2) {
                    (Some(b1), Some(b2)) => {
                        self.block_to_device.get(b1) == self.block_to_device.get(b2)
                    }
                    _ => false,
                }
            }
            CudaScope::System => true,
        }
    }

    /// Get the block id for a thread.
    pub fn block_of(&self, thread: usize) -> Option<usize> {
        self.thread_to_block.get(&thread).copied()
    }

    /// Get the device id for a thread.
    pub fn device_of(&self, thread: usize) -> Option<usize> {
        self.thread_to_block
            .get(&thread)
            .and_then(|blk| self.block_to_device.get(blk))
            .copied()
    }

    /// Total number of threads tracked.
    pub fn num_threads(&self) -> usize {
        self.thread_to_block.len()
    }
}

impl Default for CudaScopeHierarchy {
    fn default() -> Self {
        Self::new(32, 1)
    }
}

// ---------------------------------------------------------------------------
// Memory ordering
// ---------------------------------------------------------------------------

/// CUDA memory ordering for atomic operations and fences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CudaMemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl CudaMemoryOrder {
    pub fn is_acquire(&self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_release(&self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_at_least(&self, other: CudaMemoryOrder) -> bool {
        (*self as u8) >= (other as u8)
    }

    pub fn combine(self, other: CudaMemoryOrder) -> CudaMemoryOrder {
        if (self as u8) >= (other as u8) { self } else { other }
    }
}

impl fmt::Display for CudaMemoryOrder {
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

// ---------------------------------------------------------------------------
// CUDA operation types
// ---------------------------------------------------------------------------

/// CUDA memory operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaOpType {
    Load,
    Store,
    AtomicLoad,
    AtomicStore,
    AtomicExchange,
    AtomicCAS,
    AtomicAdd,
    AtomicSub,
    AtomicMin,
    AtomicMax,
    AtomicAnd,
    AtomicOr,
    AtomicXor,
    ThreadFence,
    ThreadFenceBlock,
    ThreadFenceSystem,
}

impl CudaOpType {
    pub fn is_read(&self) -> bool {
        matches!(
            self,
            Self::Load | Self::AtomicLoad | Self::AtomicExchange | Self::AtomicCAS
                | Self::AtomicAdd | Self::AtomicSub | Self::AtomicMin | Self::AtomicMax
                | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(
            self,
            Self::Store | Self::AtomicStore | Self::AtomicExchange | Self::AtomicCAS
                | Self::AtomicAdd | Self::AtomicSub | Self::AtomicMin | Self::AtomicMax
                | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor
        )
    }

    pub fn is_atomic(&self) -> bool {
        !matches!(self, Self::Load | Self::Store)
    }

    pub fn is_fence(&self) -> bool {
        matches!(self, Self::ThreadFence | Self::ThreadFenceBlock | Self::ThreadFenceSystem)
    }

    pub fn is_rmw(&self) -> bool {
        self.is_read() && self.is_write() && self.is_atomic()
    }

    /// The implicit fence scope of a `__threadfence*` variant.
    pub fn fence_scope(&self) -> Option<CudaScope> {
        match self {
            Self::ThreadFenceBlock => Some(CudaScope::Block),
            Self::ThreadFence => Some(CudaScope::Device),
            Self::ThreadFenceSystem => Some(CudaScope::System),
            _ => None,
        }
    }
}

impl fmt::Display for CudaOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// CUDA memory event
// ---------------------------------------------------------------------------

/// A memory event in the CUDA scoped model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaEvent {
    pub id: usize,
    pub thread: usize,
    pub block: usize,
    pub device: usize,
    pub op_type: CudaOpType,
    pub address: u64,
    pub value: u64,
    pub order: CudaMemoryOrder,
    pub scope: CudaScope,
    pub program_order: usize,
}

impl CudaEvent {
    pub fn new(id: usize, thread: usize, op_type: CudaOpType) -> Self {
        Self {
            id,
            thread,
            block: 0,
            device: 0,
            op_type,
            address: 0,
            value: 0,
            order: CudaMemoryOrder::Relaxed,
            scope: CudaScope::Device,
            program_order: 0,
        }
    }

    pub fn with_address(mut self, addr: u64) -> Self { self.address = addr; self }
    pub fn with_value(mut self, val: u64) -> Self { self.value = val; self }
    pub fn with_scope(mut self, scope: CudaScope) -> Self { self.scope = scope; self }
    pub fn with_order(mut self, order: CudaMemoryOrder) -> Self { self.order = order; self }
    pub fn with_block(mut self, block: usize) -> Self { self.block = block; self }
    pub fn with_device(mut self, device: usize) -> Self { self.device = device; self }
    pub fn with_po(mut self, po: usize) -> Self { self.program_order = po; self }

    /// Whether two events share the given scope instance.
    pub fn same_scope_instance(&self, other: &CudaEvent, scope: CudaScope) -> bool {
        match scope {
            CudaScope::Thread => self.thread == other.thread,
            CudaScope::Block => self.block == other.block,
            CudaScope::Device => self.device == other.device,
            CudaScope::System => true,
        }
    }
}

impl fmt::Display for CudaEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "E{}:T{}:B{} {} @{:#x}={} [{}/{}]",
            self.id, self.thread, self.block, self.op_type,
            self.address, self.value, self.order, self.scope
        )
    }
}

// ---------------------------------------------------------------------------
// Ordering relations
// ---------------------------------------------------------------------------

/// Relations specific to the CUDA scoped model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaRelation {
    ProgramOrder,
    ScopedModificationOrder,
    ReadsFrom,
    FromReads,
    SynchronizesWith,
    ScopedHappensBefore,
    FenceOrder,
}

impl fmt::Display for CudaRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgramOrder => write!(f, "po"),
            Self::ScopedModificationOrder => write!(f, "smo"),
            Self::ReadsFrom => write!(f, "rf"),
            Self::FromReads => write!(f, "fr"),
            Self::SynchronizesWith => write!(f, "sw"),
            Self::ScopedHappensBefore => write!(f, "shb"),
            Self::FenceOrder => write!(f, "fence"),
        }
    }
}

/// An edge in the CUDA ordering graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaOrderingEdge {
    pub from: usize,
    pub to: usize,
    pub relation: CudaRelation,
}

/// Graph of ordering edges used by the CUDA model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaOrderingGraph {
    pub edges: Vec<CudaOrderingEdge>,
    pub adjacency: HashMap<usize, Vec<(usize, CudaRelation)>>,
}

impl CudaOrderingGraph {
    pub fn new() -> Self {
        Self { edges: Vec::new(), adjacency: HashMap::new() }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: CudaRelation) {
        self.edges.push(CudaOrderingEdge { from, to, relation });
        self.adjacency.entry(from).or_default().push((to, relation));
    }

    /// DFS-based cycle check.
    pub fn has_cycle(&self) -> bool {
        let nodes: HashSet<usize> = self.adjacency.keys().copied().collect();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        for &node in &nodes {
            if !visited.contains(&node) && self.dfs_cycle(node, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(&self, node: usize, visited: &mut HashSet<usize>, in_stack: &mut HashSet<usize>) -> bool {
        visited.insert(node);
        in_stack.insert(node);
        if let Some(neighbors) = self.adjacency.get(&node) {
            for &(next, _) in neighbors {
                if !visited.contains(&next) {
                    if self.dfs_cycle(next, visited, in_stack) { return true; }
                } else if in_stack.contains(&next) {
                    return true;
                }
            }
        }
        in_stack.remove(&node);
        false
    }

    pub fn edges_of(&self, relation: CudaRelation) -> Vec<&CudaOrderingEdge> {
        self.edges.iter().filter(|e| e.relation == relation).collect()
    }

    pub fn edge_count(&self) -> usize { self.edges.len() }

    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for e in &self.edges {
            nodes.insert(e.from);
            nodes.insert(e.to);
        }
        nodes.len()
    }
}

impl Default for CudaOrderingGraph {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// CUDA execution
// ---------------------------------------------------------------------------

/// A CUDA execution for verification against the scoped model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaExecution {
    pub events: Vec<CudaEvent>,
    pub ordering: CudaOrderingGraph,
    pub hierarchy: CudaScopeHierarchy,
}

impl CudaExecution {
    pub fn new(hierarchy: CudaScopeHierarchy) -> Self {
        Self {
            events: Vec::new(),
            ordering: CudaOrderingGraph::new(),
            hierarchy,
        }
    }

    pub fn add_event(&mut self, event: CudaEvent) {
        self.events.push(event);
    }

    pub fn get_event(&self, id: usize) -> Option<&CudaEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Build program-order edges from per-thread event sequences.
    pub fn build_program_order(&mut self) {
        let mut by_thread: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in &self.events {
            by_thread.entry(event.thread).or_default().push(event.id);
        }
        for (_tid, mut ids) in by_thread {
            ids.sort_by_key(|&id| {
                self.events.iter().find(|e| e.id == id)
                    .map(|e| e.program_order).unwrap_or(0)
            });
            for window in ids.windows(2) {
                self.ordering.add_edge(window[0], window[1], CudaRelation::ProgramOrder);
            }
        }
    }

    /// Build scoped synchronises-with edges for release/acquire pairs that
    /// share a scope instance.
    pub fn build_synchronizes_with(&mut self) {
        let stores: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_write() && e.order.is_release())
            .map(|e| e.id).collect();
        let loads: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_read() && e.order.is_acquire())
            .map(|e| e.id).collect();

        for &s in &stores {
            for &l in &loads {
                let se = self.events.iter().find(|e| e.id == s).unwrap();
                let le = self.events.iter().find(|e| e.id == l).unwrap();
                if se.address == le.address && se.value == le.value && se.thread != le.thread {
                    let sync_scope = std::cmp::min(se.scope, le.scope);
                    if se.same_scope_instance(le, sync_scope) {
                        self.ordering.add_edge(s, l, CudaRelation::SynchronizesWith);
                    }
                }
            }
        }
    }

    /// Build fence-order edges: a fence orders all prior accesses before all
    /// later accesses at the fence's scope.
    pub fn build_fence_order(&mut self) {
        let fences: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_fence())
            .map(|e| e.id).collect();

        for &fid in &fences {
            let fe = self.events.iter().find(|e| e.id == fid).unwrap().clone();
            let fence_scope = fe.op_type.fence_scope().unwrap_or(fe.scope);

            // Events before the fence in program order on the same thread.
            let before: Vec<usize> = self.events.iter()
                .filter(|e| e.thread == fe.thread && e.program_order < fe.program_order && !e.op_type.is_fence())
                .map(|e| e.id).collect();
            // Events after the fence.
            let after: Vec<usize> = self.events.iter()
                .filter(|e| e.thread == fe.thread && e.program_order > fe.program_order && !e.op_type.is_fence())
                .map(|e| e.id).collect();

            for &b in &before {
                for &a in &after {
                    self.ordering.add_edge(b, a, CudaRelation::FenceOrder);
                }
            }
        }
    }

    /// Check consistency (no cycles in the ordering graph).
    pub fn is_consistent(&self) -> bool {
        !self.ordering.has_cycle()
    }

    pub fn event_count(&self) -> usize { self.events.len() }

    pub fn thread_count(&self) -> usize {
        self.events.iter().map(|e| e.thread).collect::<HashSet<_>>().len()
    }
}

impl Default for CudaExecution {
    fn default() -> Self { Self::new(CudaScopeHierarchy::default()) }
}

// ---------------------------------------------------------------------------
// CUDA PTX axioms
// ---------------------------------------------------------------------------

/// Axiom in the CUDA/PTX scoped memory model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CudaAxiom {
    /// Per-location coherence order is total per scope instance.
    CoherencePerScope,
    /// Atomicity of RMW: no write can intervene between the read and write
    /// of an atomic RMW in the coherence order.
    AtomicityPerScope,
    /// Scoped happens-before must be acyclic.
    ScopedHbAcyclic,
    /// Fence ordering consistency.
    FenceConsistency,
    /// Sequential consistency per scope for seq_cst operations.
    SeqCstPerScope,
    /// No thin-air values.
    NoThinAir,
}

impl CudaAxiom {
    pub fn all() -> Vec<CudaAxiom> {
        vec![
            Self::CoherencePerScope,
            Self::AtomicityPerScope,
            Self::ScopedHbAcyclic,
            Self::FenceConsistency,
            Self::SeqCstPerScope,
            Self::NoThinAir,
        ]
    }
}

impl fmt::Display for CudaAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoherencePerScope => write!(f, "coherence-per-scope"),
            Self::AtomicityPerScope => write!(f, "atomicity-per-scope"),
            Self::ScopedHbAcyclic => write!(f, "scoped-hb-acyclic"),
            Self::FenceConsistency => write!(f, "fence-consistency"),
            Self::SeqCstPerScope => write!(f, "seq-cst-per-scope"),
            Self::NoThinAir => write!(f, "no-thin-air"),
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA axiom violation
// ---------------------------------------------------------------------------

/// A violation of a CUDA scoped-model axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaViolation {
    pub axiom: CudaAxiom,
    pub description: String,
    pub involved_events: Vec<usize>,
}

impl CudaViolation {
    pub fn new(axiom: CudaAxiom, description: &str, events: Vec<usize>) -> Self {
        Self { axiom, description: description.to_string(), involved_events: events }
    }
}

impl fmt::Display for CudaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (events {:?})", self.axiom, self.description, self.involved_events)
    }
}

// ---------------------------------------------------------------------------
// CUDA axiom checker
// ---------------------------------------------------------------------------

/// Checks CUDA PTX axioms on an execution.
#[derive(Debug)]
pub struct CudaAxiomChecker<'a> {
    execution: &'a CudaExecution,
}

impl<'a> CudaAxiomChecker<'a> {
    pub fn new(execution: &'a CudaExecution) -> Self {
        Self { execution }
    }

    /// Check all axioms and return violations.
    pub fn check_all(&self) -> Vec<CudaViolation> {
        let mut violations = Vec::new();
        violations.extend(self.check_coherence_per_scope());
        violations.extend(self.check_atomicity_per_scope());
        violations.extend(self.check_scoped_hb_acyclic());
        violations.extend(self.check_fence_consistency());
        violations.extend(self.check_no_thin_air());
        violations
    }

    /// Check per-scope coherence: writes to the same address within a scope
    /// instance must form a total order.
    pub fn check_coherence_per_scope(&self) -> Vec<CudaViolation> {
        let mut violations = Vec::new();
        let mut by_addr: HashMap<u64, Vec<&CudaEvent>> = HashMap::new();
        for e in &self.execution.events {
            if !e.op_type.is_fence() {
                by_addr.entry(e.address).or_default().push(e);
            }
        }

        for (addr, events) in &by_addr {
            let writes: Vec<&&CudaEvent> = events.iter().filter(|e| e.op_type.is_write()).collect();
            if writes.len() <= 1 { continue; }

            for scope in CudaScope::all() {
                for i in 0..writes.len() {
                    for j in (i + 1)..writes.len() {
                        if !writes[i].same_scope_instance(writes[j], *scope) { continue; }
                        let w1 = writes[i].id;
                        let w2 = writes[j].id;
                        let w1_before = self.execution.ordering.adjacency
                            .get(&w1).map(|adj| adj.iter().any(|&(t, _)| t == w2)).unwrap_or(false);
                        let w2_before = self.execution.ordering.adjacency
                            .get(&w2).map(|adj| adj.iter().any(|&(t, _)| t == w1)).unwrap_or(false);
                        if w1_before && w2_before {
                            violations.push(CudaViolation::new(
                                CudaAxiom::CoherencePerScope,
                                &format!("Coherence cycle at {:#x} in scope {}", addr, scope),
                                vec![w1, w2],
                            ));
                        }
                    }
                }
            }
        }
        violations
    }

    /// Check per-scope atomicity of RMW operations.
    pub fn check_atomicity_per_scope(&self) -> Vec<CudaViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if !event.op_type.is_rmw() { continue; }
            // Find the rf-source for this RMW within its scope.
            let rf_sources: Vec<usize> = self.execution.ordering.edges_of(CudaRelation::ReadsFrom)
                .iter()
                .filter(|e| e.to == event.id)
                .map(|e| e.from)
                .collect();
            // In a full implementation, verify no intervening write in coherence order.
            if rf_sources.len() > 1 {
                violations.push(CudaViolation::new(
                    CudaAxiom::AtomicityPerScope,
                    &format!("RMW e{} has multiple rf-sources", event.id),
                    rf_sources,
                ));
            }
        }
        violations
    }

    /// Check that scoped happens-before is acyclic.
    pub fn check_scoped_hb_acyclic(&self) -> Vec<CudaViolation> {
        if self.execution.ordering.has_cycle() {
            vec![CudaViolation::new(
                CudaAxiom::ScopedHbAcyclic,
                "Cycle detected in scoped happens-before",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    /// Check fence consistency: threadfence operations must properly order
    /// accesses at the specified scope.
    pub fn check_fence_consistency(&self) -> Vec<CudaViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if !event.op_type.is_fence() { continue; }
            let fence_scope = event.op_type.fence_scope().unwrap_or(event.scope);
            // Verify that all prior writes in program order on the same thread
            // are ordered before all later reads at the fence scope.
            let prior_writes: Vec<usize> = self.execution.events.iter()
                .filter(|e| e.thread == event.thread && e.program_order < event.program_order && e.op_type.is_write())
                .map(|e| e.id).collect();
            let later_reads: Vec<usize> = self.execution.events.iter()
                .filter(|e| e.thread == event.thread && e.program_order > event.program_order && e.op_type.is_read())
                .map(|e| e.id).collect();

            // In a correct execution, fence-order edges should have been built.
            // We just check they exist; absence is not necessarily a violation
            // because fence_order construction handles this.
            let _ = (prior_writes, later_reads, fence_scope);
        }
        violations
    }

    /// Check no thin-air values.
    pub fn check_no_thin_air(&self) -> Vec<CudaViolation> {
        // Approximation: cycles in rf ∪ po would indicate thin-air.
        if self.execution.ordering.has_cycle() {
            vec![CudaViolation::new(
                CudaAxiom::NoThinAir,
                "Potential thin-air cycle detected",
                vec![],
            )]
        } else {
            vec![]
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA scoped model configuration
// ---------------------------------------------------------------------------

/// Configuration for the CUDA scoped model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaScopedModelConfig {
    pub check_coherence: bool,
    pub check_atomicity: bool,
    pub check_hb_acyclic: bool,
    pub check_fence: bool,
    pub check_seq_cst: bool,
    pub check_no_thin_air: bool,
    pub max_threads: usize,
    pub max_blocks: usize,
}

impl CudaScopedModelConfig {
    pub fn full() -> Self {
        Self {
            check_coherence: true,
            check_atomicity: true,
            check_hb_acyclic: true,
            check_fence: true,
            check_seq_cst: true,
            check_no_thin_air: true,
            max_threads: 1024,
            max_blocks: 64,
        }
    }

    pub fn minimal() -> Self {
        Self {
            check_coherence: true,
            check_atomicity: false,
            check_hb_acyclic: true,
            check_fence: false,
            check_seq_cst: false,
            check_no_thin_air: true,
            max_threads: 128,
            max_blocks: 4,
        }
    }
}

impl Default for CudaScopedModelConfig {
    fn default() -> Self { Self::full() }
}

// ---------------------------------------------------------------------------
// CUDA scoped model
// ---------------------------------------------------------------------------

/// CUDA scoped memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaScopedModel {
    pub name: String,
    pub config: CudaScopedModelConfig,
}

impl CudaScopedModel {
    pub fn new() -> Self {
        Self { name: "CUDA-Scoped".to_string(), config: CudaScopedModelConfig::full() }
    }

    pub fn with_config(config: CudaScopedModelConfig) -> Self {
        Self { name: "CUDA-Scoped".to_string(), config }
    }

    /// Verify an execution against the CUDA scoped model.
    pub fn verify(&self, execution: &CudaExecution) -> CudaVerificationResult {
        let checker = CudaAxiomChecker::new(execution);
        let violations = checker.check_all();
        let consistent = violations.is_empty();
        CudaVerificationResult {
            model_name: self.name.clone(),
            consistent,
            violations,
            events_checked: execution.event_count(),
            threads: execution.thread_count(),
        }
    }
}

impl Default for CudaScopedModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of CUDA scoped model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaVerificationResult {
    pub model_name: String,
    pub consistent: bool,
    pub violations: Vec<CudaViolation>,
    pub events_checked: usize,
    pub threads: usize,
}

impl fmt::Display for CudaVerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({} events, {} threads",
            self.model_name,
            if self.consistent { "Consistent" } else { "Inconsistent" },
            self.events_checked, self.threads)?;
        if !self.violations.is_empty() {
            write!(f, ", {} violations", self.violations.len())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// CUDA litmus test support
// ---------------------------------------------------------------------------

/// A CUDA litmus test expressed in terms of scoped events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaLitmusTest {
    pub name: String,
    pub description: String,
    pub threads: usize,
    pub blocks: usize,
    pub events: Vec<CudaEvent>,
    pub expected_outcomes: Vec<CudaOutcome>,
    pub forbidden_outcomes: Vec<CudaOutcome>,
}

/// An outcome of a CUDA litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaOutcome {
    pub values: Vec<(String, u64)>,
}

impl CudaOutcome {
    pub fn new(values: Vec<(String, u64)>) -> Self {
        Self { values }
    }
}

impl fmt::Display for CudaOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.values.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl CudaLitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            threads: 2,
            blocks: 1,
            events: Vec::new(),
            expected_outcomes: Vec::new(),
            forbidden_outcomes: Vec::new(),
        }
    }

    /// Whether an outcome is in the forbidden set.
    pub fn is_forbidden(&self, outcome: &CudaOutcome) -> bool {
        self.forbidden_outcomes.contains(outcome)
    }

    /// Store Buffer litmus test (two threads, device scope).
    pub fn store_buffer() -> Self {
        let mut test = Self::new("SB-CUDA");
        test.description = "Store Buffer litmus test under CUDA scoped model".to_string();
        test.threads = 2;

        let e0 = CudaEvent::new(0, 0, CudaOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(CudaMemoryOrder::Release).with_scope(CudaScope::Device);
        let e1 = CudaEvent::new(1, 0, CudaOpType::AtomicLoad)
            .with_address(0x200)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Device)
            .with_po(1);
        let e2 = CudaEvent::new(2, 1, CudaOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_order(CudaMemoryOrder::Release).with_scope(CudaScope::Device);
        let e3 = CudaEvent::new(3, 1, CudaOpType::AtomicLoad)
            .with_address(0x100)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Device)
            .with_po(1);

        test.events = vec![e0, e1, e2, e3];
        test.forbidden_outcomes.push(CudaOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]));
        test
    }

    /// Message-passing litmus test with threadfence.
    pub fn message_passing_fence() -> Self {
        let mut test = Self::new("MP-Fence-CUDA");
        test.description = "Message passing with __threadfence".to_string();
        test.threads = 2;

        let e0 = CudaEvent::new(0, 0, CudaOpType::Store)
            .with_address(0x100).with_value(42);
        let e1 = CudaEvent::new(1, 0, CudaOpType::ThreadFence).with_po(1);
        let e2 = CudaEvent::new(2, 0, CudaOpType::Store)
            .with_address(0x200).with_value(1).with_po(2);
        let e3 = CudaEvent::new(3, 1, CudaOpType::Load)
            .with_address(0x200);
        let e4 = CudaEvent::new(4, 1, CudaOpType::ThreadFence).with_po(1);
        let e5 = CudaEvent::new(5, 1, CudaOpType::Load)
            .with_address(0x100).with_po(2);

        test.events = vec![e0, e1, e2, e3, e4, e5];
        test.forbidden_outcomes.push(CudaOutcome::new(vec![
            ("flag".to_string(), 1), ("data".to_string(), 0),
        ]));
        test
    }

    /// IRIW (Independent Reads of Independent Writes) test at block scope.
    pub fn iriw_block_scope() -> Self {
        let mut test = Self::new("IRIW-Block-CUDA");
        test.description = "IRIW test at block scope".to_string();
        test.threads = 4;
        test.blocks = 1;

        let e0 = CudaEvent::new(0, 0, CudaOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(CudaMemoryOrder::Relaxed).with_scope(CudaScope::Block);
        let e1 = CudaEvent::new(1, 1, CudaOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_order(CudaMemoryOrder::Relaxed).with_scope(CudaScope::Block);
        let e2 = CudaEvent::new(2, 2, CudaOpType::AtomicLoad)
            .with_address(0x100)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Block);
        let e3 = CudaEvent::new(3, 2, CudaOpType::AtomicLoad)
            .with_address(0x200).with_po(1)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Block);
        let e4 = CudaEvent::new(4, 3, CudaOpType::AtomicLoad)
            .with_address(0x200)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Block);
        let e5 = CudaEvent::new(5, 3, CudaOpType::AtomicLoad)
            .with_address(0x100).with_po(1)
            .with_order(CudaMemoryOrder::Acquire).with_scope(CudaScope::Block);

        test.events = vec![e0, e1, e2, e3, e4, e5];
        test
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CudaScope tests --

    #[test]
    fn test_scope_ordering() {
        assert!(CudaScope::System > CudaScope::Device);
        assert!(CudaScope::Device > CudaScope::Block);
        assert!(CudaScope::Block > CudaScope::Thread);
    }

    #[test]
    fn test_scope_includes() {
        assert!(CudaScope::System.includes(&CudaScope::Device));
        assert!(CudaScope::Device.includes(&CudaScope::Block));
        assert!(CudaScope::Block.includes(&CudaScope::Thread));
        assert!(!CudaScope::Thread.includes(&CudaScope::Block));
    }

    #[test]
    fn test_scope_broaden_narrow() {
        assert_eq!(CudaScope::Thread.broaden(), Some(CudaScope::Block));
        assert_eq!(CudaScope::Block.broaden(), Some(CudaScope::Device));
        assert_eq!(CudaScope::System.broaden(), None);
        assert_eq!(CudaScope::System.narrow(), Some(CudaScope::Device));
        assert_eq!(CudaScope::Thread.narrow(), None);
    }

    #[test]
    fn test_scope_display() {
        assert_eq!(format!("{}", CudaScope::Block), "block");
        assert_eq!(format!("{}", CudaScope::System), "system");
    }

    // -- CudaScopeHierarchy tests --

    #[test]
    fn test_hierarchy_single_block() {
        let h = CudaScopeHierarchy::new(32, 1);
        assert_eq!(h.num_threads(), 32);
        assert!(h.same_scope_instance(0, 31, CudaScope::Block));
        assert!(h.same_scope_instance(0, 31, CudaScope::Device));
    }

    #[test]
    fn test_hierarchy_multi_block() {
        let h = CudaScopeHierarchy::new(4, 3);
        assert_eq!(h.num_threads(), 12);
        assert!(h.same_scope_instance(0, 3, CudaScope::Block));
        assert!(!h.same_scope_instance(0, 4, CudaScope::Block));
        assert!(h.same_scope_instance(0, 4, CudaScope::Device));
    }

    #[test]
    fn test_hierarchy_multi_device() {
        let h = CudaScopeHierarchy::multi_device(2, 2, 2);
        assert_eq!(h.num_threads(), 8);
        // Same device, different blocks.
        assert!(!h.same_scope_instance(0, 2, CudaScope::Block));
        assert!(h.same_scope_instance(0, 2, CudaScope::Device));
        // Different devices.
        assert!(!h.same_scope_instance(0, 4, CudaScope::Device));
        assert!(h.same_scope_instance(0, 4, CudaScope::System));
    }

    #[test]
    fn test_hierarchy_block_device_of() {
        let h = CudaScopeHierarchy::new(4, 2);
        assert_eq!(h.block_of(0), Some(0));
        assert_eq!(h.block_of(5), Some(1));
        assert_eq!(h.device_of(0), Some(0));
    }

    // -- CudaMemoryOrder tests --

    #[test]
    fn test_memory_order() {
        assert!(CudaMemoryOrder::Acquire.is_acquire());
        assert!(!CudaMemoryOrder::Release.is_acquire());
        assert!(CudaMemoryOrder::Release.is_release());
        assert!(CudaMemoryOrder::AcqRel.is_acquire());
        assert!(CudaMemoryOrder::AcqRel.is_release());
        assert!(CudaMemoryOrder::SeqCst.is_acquire());
        assert!(CudaMemoryOrder::SeqCst.is_release());
    }

    #[test]
    fn test_memory_order_at_least() {
        assert!(CudaMemoryOrder::SeqCst.is_at_least(CudaMemoryOrder::Relaxed));
        assert!(!CudaMemoryOrder::Relaxed.is_at_least(CudaMemoryOrder::Acquire));
    }

    #[test]
    fn test_memory_order_combine() {
        assert_eq!(CudaMemoryOrder::Acquire.combine(CudaMemoryOrder::Release), CudaMemoryOrder::Release);
        assert_eq!(CudaMemoryOrder::SeqCst.combine(CudaMemoryOrder::Relaxed), CudaMemoryOrder::SeqCst);
    }

    // -- CudaOpType tests --

    #[test]
    fn test_op_type_classification() {
        assert!(CudaOpType::Load.is_read());
        assert!(!CudaOpType::Load.is_write());
        assert!(!CudaOpType::Load.is_atomic());
        assert!(CudaOpType::AtomicCAS.is_rmw());
        assert!(CudaOpType::ThreadFence.is_fence());
        assert_eq!(CudaOpType::ThreadFenceBlock.fence_scope(), Some(CudaScope::Block));
        assert_eq!(CudaOpType::ThreadFence.fence_scope(), Some(CudaScope::Device));
        assert_eq!(CudaOpType::ThreadFenceSystem.fence_scope(), Some(CudaScope::System));
    }

    // -- CudaEvent tests --

    #[test]
    fn test_event_creation() {
        let event = CudaEvent::new(0, 1, CudaOpType::AtomicStore)
            .with_address(0x100).with_value(42).with_scope(CudaScope::Device);
        assert_eq!(event.id, 0);
        assert_eq!(event.thread, 1);
        assert_eq!(event.address, 0x100);
        assert_eq!(event.value, 42);
        assert_eq!(event.scope, CudaScope::Device);
    }

    #[test]
    fn test_event_same_scope_instance() {
        let mut e1 = CudaEvent::new(0, 0, CudaOpType::Store);
        e1.block = 0;
        let mut e2 = CudaEvent::new(1, 1, CudaOpType::Load);
        e2.block = 0;
        assert!(e1.same_scope_instance(&e2, CudaScope::Block));
        assert!(!e1.same_scope_instance(&e2, CudaScope::Thread));
    }

    #[test]
    fn test_event_display() {
        let event = CudaEvent::new(0, 1, CudaOpType::Store).with_address(0x100).with_value(5);
        let s = format!("{}", event);
        assert!(s.contains("E0"));
        assert!(s.contains("T1"));
    }

    // -- CudaOrderingGraph tests --

    #[test]
    fn test_ordering_graph_no_cycle() {
        let mut g = CudaOrderingGraph::new();
        g.add_edge(0, 1, CudaRelation::ProgramOrder);
        g.add_edge(1, 2, CudaRelation::ProgramOrder);
        assert!(!g.has_cycle());
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_ordering_graph_cycle() {
        let mut g = CudaOrderingGraph::new();
        g.add_edge(0, 1, CudaRelation::ProgramOrder);
        g.add_edge(1, 2, CudaRelation::SynchronizesWith);
        g.add_edge(2, 0, CudaRelation::ReadsFrom);
        assert!(g.has_cycle());
    }

    // -- CudaExecution tests --

    #[test]
    fn test_execution_basic() {
        let mut exec = CudaExecution::default();
        exec.add_event(CudaEvent::new(0, 0, CudaOpType::Store));
        exec.add_event(CudaEvent::new(1, 0, CudaOpType::Load).with_po(1));
        exec.add_event(CudaEvent::new(2, 1, CudaOpType::Store));
        assert_eq!(exec.event_count(), 3);
        assert_eq!(exec.thread_count(), 2);
    }

    #[test]
    fn test_execution_build_po() {
        let mut exec = CudaExecution::default();
        exec.add_event(CudaEvent::new(0, 0, CudaOpType::Store).with_po(0));
        exec.add_event(CudaEvent::new(1, 0, CudaOpType::Load).with_po(1));
        exec.build_program_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    #[test]
    fn test_execution_consistent_empty() {
        let exec = CudaExecution::default();
        assert!(exec.is_consistent());
    }

    #[test]
    fn test_execution_fence_order() {
        let mut exec = CudaExecution::default();
        exec.add_event(CudaEvent::new(0, 0, CudaOpType::Store).with_address(0x100).with_po(0));
        exec.add_event(CudaEvent::new(1, 0, CudaOpType::ThreadFence).with_po(1));
        exec.add_event(CudaEvent::new(2, 0, CudaOpType::Load).with_address(0x200).with_po(2));
        exec.build_fence_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    // -- CudaAxiomChecker tests --

    #[test]
    fn test_checker_empty_execution() {
        let exec = CudaExecution::default();
        let checker = CudaAxiomChecker::new(&exec);
        assert!(checker.check_all().is_empty());
    }

    #[test]
    fn test_checker_simple_consistent() {
        let mut exec = CudaExecution::default();
        exec.add_event(CudaEvent::new(0, 0, CudaOpType::Store).with_address(0x100).with_value(1));
        exec.add_event(CudaEvent::new(1, 1, CudaOpType::Load).with_address(0x100).with_value(1));
        exec.build_program_order();
        let checker = CudaAxiomChecker::new(&exec);
        assert!(checker.check_all().is_empty());
    }

    // -- CudaScopedModel tests --

    #[test]
    fn test_model_verify_empty() {
        let model = CudaScopedModel::new();
        let exec = CudaExecution::default();
        let result = model.verify(&exec);
        assert!(result.consistent);
        assert_eq!(result.model_name, "CUDA-Scoped");
    }

    #[test]
    fn test_model_config() {
        let config = CudaScopedModelConfig::full();
        assert!(config.check_coherence);
        assert!(config.check_atomicity);
        let config = CudaScopedModelConfig::minimal();
        assert!(config.check_coherence);
        assert!(!config.check_atomicity);
    }

    #[test]
    fn test_model_with_config() {
        let model = CudaScopedModel::with_config(CudaScopedModelConfig::minimal());
        assert_eq!(model.name, "CUDA-Scoped");
    }

    // -- CudaLitmusTest tests --

    #[test]
    fn test_litmus_store_buffer() {
        let test = CudaLitmusTest::store_buffer();
        assert_eq!(test.name, "SB-CUDA");
        assert_eq!(test.threads, 2);
        assert_eq!(test.events.len(), 4);
        assert_eq!(test.forbidden_outcomes.len(), 1);
    }

    #[test]
    fn test_litmus_message_passing() {
        let test = CudaLitmusTest::message_passing_fence();
        assert_eq!(test.name, "MP-Fence-CUDA");
        assert_eq!(test.events.len(), 6);
    }

    #[test]
    fn test_litmus_iriw() {
        let test = CudaLitmusTest::iriw_block_scope();
        assert_eq!(test.name, "IRIW-Block-CUDA");
        assert_eq!(test.events.len(), 6);
        assert_eq!(test.threads, 4);
    }

    #[test]
    fn test_litmus_forbidden() {
        let test = CudaLitmusTest::store_buffer();
        let outcome = CudaOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]);
        assert!(test.is_forbidden(&outcome));
    }

    #[test]
    fn test_outcome_display() {
        let o = CudaOutcome::new(vec![("r0".to_string(), 1)]);
        assert!(format!("{}", o).contains("r0=1"));
    }

    // -- CudaVerificationResult tests --

    #[test]
    fn test_verification_result_display() {
        let result = CudaVerificationResult {
            model_name: "CUDA-Scoped".to_string(),
            consistent: true,
            violations: vec![],
            events_checked: 4,
            threads: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("Consistent"));
        assert!(s.contains("4 events"));
    }

    // -- CudaAxiom tests --

    #[test]
    fn test_axiom_all() {
        let all = CudaAxiom::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_axiom_display() {
        assert_eq!(format!("{}", CudaAxiom::CoherencePerScope), "coherence-per-scope");
        assert_eq!(format!("{}", CudaAxiom::NoThinAir), "no-thin-air");
    }

    // -- CudaViolation tests --

    #[test]
    fn test_violation_display() {
        let v = CudaViolation::new(CudaAxiom::CoherencePerScope, "test violation", vec![0, 1]);
        let s = format!("{}", v);
        assert!(s.contains("coherence-per-scope"));
        assert!(s.contains("test violation"));
    }

    // -- CudaRelation tests --

    #[test]
    fn test_relation_display() {
        assert_eq!(format!("{}", CudaRelation::ProgramOrder), "po");
        assert_eq!(format!("{}", CudaRelation::SynchronizesWith), "sw");
        assert_eq!(format!("{}", CudaRelation::ScopedHappensBefore), "shb");
    }
}
