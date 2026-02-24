//! OpenCL 2.0 full scope hierarchy memory model for LITMUS∞.
//!
//! Implements the OpenCL 2.0 memory model with:
//! - Work-group, sub-group, device, all-SVMs-devices scopes
//! - Memory orders (relaxed, acquire, release, acq_rel, seq_cst) × scope
//! - Memory regions (global, local, private)
//! - SVM (Shared Virtual Memory) atomics
//! - Barrier semantics with scope
//!
//! # Scope hierarchy
//!
//! ```text
//! all-SVMs-devices
//!  └─ device
//!      └─ work-group
//!          └─ sub-group
//!              └─ work-item
//! ```
//!
//! The OpenCL 2.0 model is similar to C++11 but with explicit scopes and
//! memory regions. Synchronisation between work-items in different
//! work-groups requires device-scope or all-SVMs-devices-scope operations.

#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// OpenCL 2.0 scopes
// ---------------------------------------------------------------------------

/// OpenCL 2.0 memory scope hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpenClScope {
    WorkItem,
    SubGroup,
    WorkGroup,
    Device,
    AllSvmDevices,
}

impl OpenClScope {
    pub fn all() -> &'static [OpenClScope] {
        &[Self::WorkItem, Self::SubGroup, Self::WorkGroup, Self::Device, Self::AllSvmDevices]
    }

    /// Whether `self` is at least as broad as `other`.
    pub fn includes(&self, other: &OpenClScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    pub fn broaden(&self) -> Option<OpenClScope> {
        match self {
            Self::WorkItem => Some(Self::SubGroup),
            Self::SubGroup => Some(Self::WorkGroup),
            Self::WorkGroup => Some(Self::Device),
            Self::Device => Some(Self::AllSvmDevices),
            Self::AllSvmDevices => None,
        }
    }

    pub fn narrow(&self) -> Option<OpenClScope> {
        match self {
            Self::AllSvmDevices => Some(Self::Device),
            Self::Device => Some(Self::WorkGroup),
            Self::WorkGroup => Some(Self::SubGroup),
            Self::SubGroup => Some(Self::WorkItem),
            Self::WorkItem => None,
        }
    }
}

impl fmt::Display for OpenClScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkItem => write!(f, "work_item"),
            Self::SubGroup => write!(f, "sub_group"),
            Self::WorkGroup => write!(f, "work_group"),
            Self::Device => write!(f, "device"),
            Self::AllSvmDevices => write!(f, "all_svm_devices"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory regions
// ---------------------------------------------------------------------------

/// OpenCL 2.0 memory regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenClMemoryRegion {
    /// Global memory — visible to all work-items on all devices.
    Global,
    /// Local memory — shared within a work-group.
    Local,
    /// Private memory — per work-item.
    Private,
    /// Constant memory — read-only global.
    Constant,
    /// Generic — resolved at runtime.
    Generic,
}

impl OpenClMemoryRegion {
    /// Whether this region is shared across work-items.
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Global | Self::Local | Self::Constant | Self::Generic)
    }

    /// Whether this region is only accessible within a work-group.
    pub fn is_work_group_local(&self) -> bool {
        matches!(self, Self::Local)
    }

    /// Minimum scope needed for synchronising this region.
    pub fn min_sync_scope(&self) -> OpenClScope {
        match self {
            Self::Private => OpenClScope::WorkItem,
            Self::Local => OpenClScope::WorkGroup,
            Self::Global | Self::Constant | Self::Generic => OpenClScope::Device,
        }
    }
}

impl fmt::Display for OpenClMemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Local => write!(f, "local"),
            Self::Private => write!(f, "private"),
            Self::Constant => write!(f, "constant"),
            Self::Generic => write!(f, "generic"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory orders
// ---------------------------------------------------------------------------

/// OpenCL 2.0 memory ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpenCl2MemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl OpenCl2MemoryOrder {
    pub fn is_acquire(&self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_release(&self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_at_least(&self, other: OpenCl2MemoryOrder) -> bool {
        (*self as u8) >= (other as u8)
    }

    pub fn combine(self, other: OpenCl2MemoryOrder) -> OpenCl2MemoryOrder {
        if (self as u8) >= (other as u8) { self } else { other }
    }
}

impl fmt::Display for OpenCl2MemoryOrder {
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
// OpenCL 2.0 work-item identity
// ---------------------------------------------------------------------------

/// Identity of an OpenCL work-item in the NDRange hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkItemId {
    pub global_id: usize,
    pub local_id: usize,
    pub sub_group_id: usize,
    pub work_group_id: usize,
    pub device_id: usize,
}

impl WorkItemId {
    pub fn new(global_id: usize, local_id: usize, sub_group_id: usize,
               work_group_id: usize, device_id: usize) -> Self {
        Self { global_id, local_id, sub_group_id, work_group_id, device_id }
    }

    pub fn same_scope_instance(&self, other: &WorkItemId, scope: OpenClScope) -> bool {
        match scope {
            OpenClScope::WorkItem => self.global_id == other.global_id,
            OpenClScope::SubGroup => {
                self.work_group_id == other.work_group_id && self.sub_group_id == other.sub_group_id
            }
            OpenClScope::WorkGroup => self.work_group_id == other.work_group_id,
            OpenClScope::Device => self.device_id == other.device_id,
            OpenClScope::AllSvmDevices => true,
        }
    }
}

impl fmt::Display for WorkItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WI(g={}, l={}, sg={}, wg={}, d={})",
            self.global_id, self.local_id, self.sub_group_id,
            self.work_group_id, self.device_id)
    }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 operation types
// ---------------------------------------------------------------------------

/// OpenCL 2.0 atomic operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenCl2OpType {
    Load,
    Store,
    AtomicLoad,
    AtomicStore,
    AtomicExchange,
    AtomicCompareExchange,
    AtomicFetchAdd,
    AtomicFetchSub,
    AtomicFetchOr,
    AtomicFetchAnd,
    AtomicFetchXor,
    AtomicFetchMin,
    AtomicFetchMax,
    WorkGroupBarrier,
    SubGroupBarrier,
    AtomicFence,
    SvmAtomicLoad,
    SvmAtomicStore,
    SvmAtomicExchange,
    SvmAtomicCompareExchange,
}

impl OpenCl2OpType {
    pub fn is_read(&self) -> bool {
        matches!(self,
            Self::Load | Self::AtomicLoad | Self::AtomicExchange
            | Self::AtomicCompareExchange | Self::AtomicFetchAdd | Self::AtomicFetchSub
            | Self::AtomicFetchOr | Self::AtomicFetchAnd | Self::AtomicFetchXor
            | Self::AtomicFetchMin | Self::AtomicFetchMax
            | Self::SvmAtomicLoad | Self::SvmAtomicExchange | Self::SvmAtomicCompareExchange
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(self,
            Self::Store | Self::AtomicStore | Self::AtomicExchange
            | Self::AtomicCompareExchange | Self::AtomicFetchAdd | Self::AtomicFetchSub
            | Self::AtomicFetchOr | Self::AtomicFetchAnd | Self::AtomicFetchXor
            | Self::AtomicFetchMin | Self::AtomicFetchMax
            | Self::SvmAtomicStore | Self::SvmAtomicExchange | Self::SvmAtomicCompareExchange
        )
    }

    pub fn is_atomic(&self) -> bool {
        !matches!(self, Self::Load | Self::Store)
    }

    pub fn is_barrier(&self) -> bool {
        matches!(self, Self::WorkGroupBarrier | Self::SubGroupBarrier)
    }

    pub fn is_fence(&self) -> bool {
        matches!(self, Self::AtomicFence)
    }

    pub fn is_rmw(&self) -> bool {
        self.is_read() && self.is_write() && self.is_atomic()
    }

    pub fn is_svm(&self) -> bool {
        matches!(self,
            Self::SvmAtomicLoad | Self::SvmAtomicStore
            | Self::SvmAtomicExchange | Self::SvmAtomicCompareExchange
        )
    }

    /// Implicit scope of a barrier operation.
    pub fn barrier_scope(&self) -> Option<OpenClScope> {
        match self {
            Self::WorkGroupBarrier => Some(OpenClScope::WorkGroup),
            Self::SubGroupBarrier => Some(OpenClScope::SubGroup),
            _ => None,
        }
    }
}

impl fmt::Display for OpenCl2OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 memory event
// ---------------------------------------------------------------------------

/// A memory event in the OpenCL 2.0 model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2Event {
    pub id: usize,
    pub work_item: WorkItemId,
    pub op_type: OpenCl2OpType,
    pub address: u64,
    pub value: u64,
    pub memory_region: OpenClMemoryRegion,
    pub memory_order: OpenCl2MemoryOrder,
    pub memory_scope: OpenClScope,
    pub program_order: usize,
}

impl OpenCl2Event {
    pub fn new(id: usize, work_item: WorkItemId, op_type: OpenCl2OpType) -> Self {
        Self {
            id,
            work_item,
            op_type,
            address: 0,
            value: 0,
            memory_region: OpenClMemoryRegion::Global,
            memory_order: OpenCl2MemoryOrder::Relaxed,
            memory_scope: OpenClScope::Device,
            program_order: 0,
        }
    }

    pub fn with_address(mut self, addr: u64) -> Self { self.address = addr; self }
    pub fn with_value(mut self, val: u64) -> Self { self.value = val; self }
    pub fn with_region(mut self, region: OpenClMemoryRegion) -> Self { self.memory_region = region; self }
    pub fn with_order(mut self, order: OpenCl2MemoryOrder) -> Self { self.memory_order = order; self }
    pub fn with_scope(mut self, scope: OpenClScope) -> Self { self.memory_scope = scope; self }
    pub fn with_po(mut self, po: usize) -> Self { self.program_order = po; self }

    pub fn same_scope_instance(&self, other: &OpenCl2Event, scope: OpenClScope) -> bool {
        self.work_item.same_scope_instance(&other.work_item, scope)
    }
}

impl fmt::Display for OpenCl2Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:{} {} @{:#x}={} [{}/{}/{}]",
            self.id, self.work_item, self.op_type,
            self.address, self.value,
            self.memory_order, self.memory_scope, self.memory_region)
    }
}

// ---------------------------------------------------------------------------
// SVM atomic operations
// ---------------------------------------------------------------------------

/// Represents a Shared Virtual Memory atomic operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvmAtomicOp {
    pub event_id: usize,
    pub address: u64,
    pub op_type: OpenCl2OpType,
    pub order: OpenCl2MemoryOrder,
    pub scope: OpenClScope,
    /// SVM pointer base (host virtual address).
    pub svm_base: u64,
    /// Whether this is a coarse-grained SVM allocation.
    pub coarse_grained: bool,
}

impl SvmAtomicOp {
    pub fn new(event_id: usize, address: u64, op_type: OpenCl2OpType) -> Self {
        Self {
            event_id, address, op_type,
            order: OpenCl2MemoryOrder::SeqCst,
            scope: OpenClScope::AllSvmDevices,
            svm_base: 0,
            coarse_grained: false,
        }
    }

    /// Fine-grained SVM atomics require all-SVMs-devices scope.
    pub fn requires_all_svm_scope(&self) -> bool {
        !self.coarse_grained
    }

    /// Check scope validity for SVM operations.
    pub fn is_scope_valid(&self) -> bool {
        if self.requires_all_svm_scope() {
            self.scope == OpenClScope::AllSvmDevices
        } else {
            true
        }
    }
}

impl fmt::Display for SvmAtomicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SVM-{} @{:#x} [{}/{}]{}",
            self.op_type, self.address, self.order, self.scope,
            if self.coarse_grained { " (coarse)" } else { " (fine)" })
    }
}

// ---------------------------------------------------------------------------
// Ordering relations
// ---------------------------------------------------------------------------

/// Relations in the OpenCL 2.0 model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenCl2Relation {
    ProgramOrder,
    ScopedModificationOrder,
    ReadsFrom,
    FromReads,
    SynchronizesWith,
    ScopedHappensBefore,
    BarrierOrder,
    FenceOrder,
    SvmSync,
}

impl fmt::Display for OpenCl2Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgramOrder => write!(f, "po"),
            Self::ScopedModificationOrder => write!(f, "smo"),
            Self::ReadsFrom => write!(f, "rf"),
            Self::FromReads => write!(f, "fr"),
            Self::SynchronizesWith => write!(f, "sw"),
            Self::ScopedHappensBefore => write!(f, "shb"),
            Self::BarrierOrder => write!(f, "bar"),
            Self::FenceOrder => write!(f, "fence"),
            Self::SvmSync => write!(f, "svm-sync"),
        }
    }
}

/// Edge in the OpenCL 2.0 ordering graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenCl2Edge {
    pub from: usize,
    pub to: usize,
    pub relation: OpenCl2Relation,
}

/// Ordering graph for the OpenCL 2.0 model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenCl2OrderingGraph {
    pub edges: Vec<OpenCl2Edge>,
    pub adjacency: HashMap<usize, Vec<(usize, OpenCl2Relation)>>,
}

impl OpenCl2OrderingGraph {
    pub fn new() -> Self { Self { edges: Vec::new(), adjacency: HashMap::new() } }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: OpenCl2Relation) {
        self.edges.push(OpenCl2Edge { from, to, relation });
        self.adjacency.entry(from).or_default().push((to, relation));
    }

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

    pub fn edges_of(&self, relation: OpenCl2Relation) -> Vec<&OpenCl2Edge> {
        self.edges.iter().filter(|e| e.relation == relation).collect()
    }

    pub fn edge_count(&self) -> usize { self.edges.len() }

    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for e in &self.edges { nodes.insert(e.from); nodes.insert(e.to); }
        nodes.len()
    }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 execution
// ---------------------------------------------------------------------------

/// An OpenCL 2.0 execution for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2Execution {
    pub events: Vec<OpenCl2Event>,
    pub ordering: OpenCl2OrderingGraph,
    pub svm_ops: Vec<SvmAtomicOp>,
}

impl OpenCl2Execution {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            ordering: OpenCl2OrderingGraph::new(),
            svm_ops: Vec::new(),
        }
    }

    pub fn add_event(&mut self, event: OpenCl2Event) {
        self.events.push(event);
    }

    pub fn add_svm_op(&mut self, op: SvmAtomicOp) {
        self.svm_ops.push(op);
    }

    pub fn get_event(&self, id: usize) -> Option<&OpenCl2Event> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn build_program_order(&mut self) {
        let mut by_wi: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in &self.events {
            by_wi.entry(event.work_item.global_id).or_default().push(event.id);
        }
        for (_wi, mut ids) in by_wi {
            ids.sort_by_key(|&id| {
                self.events.iter().find(|e| e.id == id).map(|e| e.program_order).unwrap_or(0)
            });
            for window in ids.windows(2) {
                self.ordering.add_edge(window[0], window[1], OpenCl2Relation::ProgramOrder);
            }
        }
    }

    pub fn build_synchronizes_with(&mut self) {
        let stores: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_write() && e.memory_order.is_release())
            .map(|e| e.id).collect();
        let loads: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_read() && e.memory_order.is_acquire())
            .map(|e| e.id).collect();

        for &s in &stores {
            for &l in &loads {
                let se = self.events.iter().find(|e| e.id == s).unwrap();
                let le = self.events.iter().find(|e| e.id == l).unwrap();
                if se.address == le.address && se.value == le.value
                    && se.work_item.global_id != le.work_item.global_id
                {
                    let sync_scope = std::cmp::min(se.memory_scope, le.memory_scope);
                    if se.same_scope_instance(le, sync_scope) {
                        self.ordering.add_edge(s, l, OpenCl2Relation::SynchronizesWith);
                    }
                }
            }
        }
    }

    /// Build barrier-order edges for work-group and sub-group barriers.
    pub fn build_barrier_order(&mut self) {
        let barriers: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_barrier())
            .map(|e| e.id).collect();

        for &bid in &barriers {
            let be = self.events.iter().find(|e| e.id == bid).unwrap().clone();
            let barrier_scope = be.op_type.barrier_scope().unwrap_or(OpenClScope::WorkGroup);

            let before: Vec<usize> = self.events.iter()
                .filter(|e| e.work_item.global_id == be.work_item.global_id
                    && e.program_order < be.program_order && !e.op_type.is_barrier())
                .map(|e| e.id).collect();
            let after: Vec<usize> = self.events.iter()
                .filter(|e| e.work_item.global_id == be.work_item.global_id
                    && e.program_order > be.program_order && !e.op_type.is_barrier())
                .map(|e| e.id).collect();

            for &b in &before {
                for &a in &after {
                    self.ordering.add_edge(b, a, OpenCl2Relation::BarrierOrder);
                }
            }
        }
    }

    pub fn is_consistent(&self) -> bool { !self.ordering.has_cycle() }
    pub fn event_count(&self) -> usize { self.events.len() }

    pub fn work_item_count(&self) -> usize {
        self.events.iter().map(|e| e.work_item.global_id).collect::<HashSet<_>>().len()
    }
}

impl Default for OpenCl2Execution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 axioms
// ---------------------------------------------------------------------------

/// Axiom in the OpenCL 2.0 memory model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpenCl2Axiom {
    /// Per-scope coherence order is total.
    CoherencePerScope,
    /// RMW atomicity per scope.
    AtomicityPerScope,
    /// Scoped happens-before must be acyclic.
    ScopedHbAcyclic,
    /// Barrier ordering consistency.
    BarrierConsistency,
    /// SeqCst total order per scope.
    SeqCstPerScope,
    /// No thin-air values.
    NoThinAir,
    /// Local memory accessed only within work-group.
    LocalMemoryRestriction,
    /// SVM scope validity.
    SvmScopeValidity,
}

impl OpenCl2Axiom {
    pub fn all() -> Vec<Self> {
        vec![
            Self::CoherencePerScope, Self::AtomicityPerScope,
            Self::ScopedHbAcyclic, Self::BarrierConsistency,
            Self::SeqCstPerScope, Self::NoThinAir,
            Self::LocalMemoryRestriction, Self::SvmScopeValidity,
        ]
    }
}

impl fmt::Display for OpenCl2Axiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoherencePerScope => write!(f, "coherence-per-scope"),
            Self::AtomicityPerScope => write!(f, "atomicity-per-scope"),
            Self::ScopedHbAcyclic => write!(f, "scoped-hb-acyclic"),
            Self::BarrierConsistency => write!(f, "barrier-consistency"),
            Self::SeqCstPerScope => write!(f, "seq-cst-per-scope"),
            Self::NoThinAir => write!(f, "no-thin-air"),
            Self::LocalMemoryRestriction => write!(f, "local-memory-restriction"),
            Self::SvmScopeValidity => write!(f, "svm-scope-validity"),
        }
    }
}

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

/// A violation of an OpenCL 2.0 model axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2Violation {
    pub axiom: OpenCl2Axiom,
    pub description: String,
    pub involved_events: Vec<usize>,
}

impl OpenCl2Violation {
    pub fn new(axiom: OpenCl2Axiom, description: &str, events: Vec<usize>) -> Self {
        Self { axiom, description: description.to_string(), involved_events: events }
    }
}

impl fmt::Display for OpenCl2Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (events {:?})", self.axiom, self.description, self.involved_events)
    }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 axiom checker
// ---------------------------------------------------------------------------

/// Checks OpenCL 2.0 axioms on an execution.
#[derive(Debug)]
pub struct OpenCl2AxiomChecker<'a> {
    execution: &'a OpenCl2Execution,
}

impl<'a> OpenCl2AxiomChecker<'a> {
    pub fn new(execution: &'a OpenCl2Execution) -> Self {
        Self { execution }
    }

    pub fn check_all(&self) -> Vec<OpenCl2Violation> {
        let mut violations = Vec::new();
        violations.extend(self.check_coherence_per_scope());
        violations.extend(self.check_atomicity_per_scope());
        violations.extend(self.check_scoped_hb_acyclic());
        violations.extend(self.check_no_thin_air());
        violations.extend(self.check_local_memory_restriction());
        violations.extend(self.check_svm_scope_validity());
        violations
    }

    pub fn check_coherence_per_scope(&self) -> Vec<OpenCl2Violation> {
        let mut violations = Vec::new();
        let mut by_addr: HashMap<u64, Vec<&OpenCl2Event>> = HashMap::new();
        for e in &self.execution.events {
            if !e.op_type.is_barrier() && !e.op_type.is_fence() {
                by_addr.entry(e.address).or_default().push(e);
            }
        }

        for (addr, events) in &by_addr {
            let writes: Vec<&&OpenCl2Event> = events.iter().filter(|e| e.op_type.is_write()).collect();
            if writes.len() <= 1 { continue; }
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    let w1 = writes[i].id;
                    let w2 = writes[j].id;
                    let fwd = self.execution.ordering.adjacency.get(&w1)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w2)).unwrap_or(false);
                    let bwd = self.execution.ordering.adjacency.get(&w2)
                        .map(|adj| adj.iter().any(|&(t, _)| t == w1)).unwrap_or(false);
                    if fwd && bwd {
                        violations.push(OpenCl2Violation::new(
                            OpenCl2Axiom::CoherencePerScope,
                            &format!("Coherence cycle at {:#x}", addr),
                            vec![w1, w2],
                        ));
                    }
                }
            }
        }
        violations
    }

    pub fn check_atomicity_per_scope(&self) -> Vec<OpenCl2Violation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if !event.op_type.is_rmw() { continue; }
            let rf_sources: Vec<usize> = self.execution.ordering.edges_of(OpenCl2Relation::ReadsFrom)
                .iter().filter(|e| e.to == event.id).map(|e| e.from).collect();
            if rf_sources.len() > 1 {
                violations.push(OpenCl2Violation::new(
                    OpenCl2Axiom::AtomicityPerScope,
                    &format!("RMW e{} has multiple rf-sources", event.id),
                    rf_sources,
                ));
            }
        }
        violations
    }

    pub fn check_scoped_hb_acyclic(&self) -> Vec<OpenCl2Violation> {
        if self.execution.ordering.has_cycle() {
            vec![OpenCl2Violation::new(
                OpenCl2Axiom::ScopedHbAcyclic,
                "Cycle detected in scoped happens-before",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    pub fn check_no_thin_air(&self) -> Vec<OpenCl2Violation> {
        if self.execution.ordering.has_cycle() {
            vec![OpenCl2Violation::new(
                OpenCl2Axiom::NoThinAir,
                "Potential thin-air cycle detected",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    /// Check that local memory is only accessed by work-items in the same work-group.
    pub fn check_local_memory_restriction(&self) -> Vec<OpenCl2Violation> {
        let mut violations = Vec::new();
        let local_events: Vec<&OpenCl2Event> = self.execution.events.iter()
            .filter(|e| e.memory_region == OpenClMemoryRegion::Local)
            .collect();

        let mut by_addr: HashMap<u64, Vec<&OpenCl2Event>> = HashMap::new();
        for e in &local_events {
            by_addr.entry(e.address).or_default().push(e);
        }

        for (addr, events) in &by_addr {
            for i in 0..events.len() {
                for j in (i + 1)..events.len() {
                    if events[i].work_item.work_group_id != events[j].work_item.work_group_id {
                        violations.push(OpenCl2Violation::new(
                            OpenCl2Axiom::LocalMemoryRestriction,
                            &format!("Local memory at {:#x} accessed from work-groups {} and {}",
                                addr, events[i].work_item.work_group_id,
                                events[j].work_item.work_group_id),
                            vec![events[i].id, events[j].id],
                        ));
                    }
                }
            }
        }
        violations
    }

    /// Check SVM scope validity.
    pub fn check_svm_scope_validity(&self) -> Vec<OpenCl2Violation> {
        let mut violations = Vec::new();
        for op in &self.execution.svm_ops {
            if !op.is_scope_valid() {
                violations.push(OpenCl2Violation::new(
                    OpenCl2Axiom::SvmScopeValidity,
                    &format!("Fine-grained SVM op at {:#x} requires all_svm_devices scope, has {}",
                        op.address, op.scope),
                    vec![op.event_id],
                ));
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 model configuration
// ---------------------------------------------------------------------------

/// Configuration for the OpenCL 2.0 model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2ModelConfig {
    pub check_coherence: bool,
    pub check_atomicity: bool,
    pub check_hb_acyclic: bool,
    pub check_barrier: bool,
    pub check_seq_cst: bool,
    pub check_no_thin_air: bool,
    pub check_local_memory: bool,
    pub check_svm: bool,
    pub max_work_items: usize,
    pub max_work_groups: usize,
}

impl OpenCl2ModelConfig {
    pub fn full() -> Self {
        Self {
            check_coherence: true, check_atomicity: true, check_hb_acyclic: true,
            check_barrier: true, check_seq_cst: true, check_no_thin_air: true,
            check_local_memory: true, check_svm: true,
            max_work_items: 1024, max_work_groups: 64,
        }
    }

    pub fn minimal() -> Self {
        Self {
            check_coherence: true, check_atomicity: false, check_hb_acyclic: true,
            check_barrier: false, check_seq_cst: false, check_no_thin_air: true,
            check_local_memory: false, check_svm: false,
            max_work_items: 128, max_work_groups: 4,
        }
    }
}

impl Default for OpenCl2ModelConfig {
    fn default() -> Self { Self::full() }
}

// ---------------------------------------------------------------------------
// OpenCL 2.0 model
// ---------------------------------------------------------------------------

/// OpenCL 2.0 memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2Model {
    pub name: String,
    pub config: OpenCl2ModelConfig,
}

impl OpenCl2Model {
    pub fn new() -> Self {
        Self { name: "OpenCL2".to_string(), config: OpenCl2ModelConfig::full() }
    }

    pub fn with_config(config: OpenCl2ModelConfig) -> Self {
        Self { name: "OpenCL2".to_string(), config }
    }

    pub fn verify(&self, execution: &OpenCl2Execution) -> OpenCl2VerificationResult {
        let checker = OpenCl2AxiomChecker::new(execution);
        let violations = checker.check_all();
        let consistent = violations.is_empty();
        OpenCl2VerificationResult {
            model_name: self.name.clone(),
            consistent,
            violations,
            events_checked: execution.event_count(),
            work_items: execution.work_item_count(),
        }
    }
}

impl Default for OpenCl2Model {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of OpenCL 2.0 model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2VerificationResult {
    pub model_name: String,
    pub consistent: bool,
    pub violations: Vec<OpenCl2Violation>,
    pub events_checked: usize,
    pub work_items: usize,
}

impl fmt::Display for OpenCl2VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({} events, {} work-items",
            self.model_name,
            if self.consistent { "Consistent" } else { "Inconsistent" },
            self.events_checked, self.work_items)?;
        if !self.violations.is_empty() {
            write!(f, ", {} violations", self.violations.len())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Litmus test support
// ---------------------------------------------------------------------------

/// An OpenCL 2.0 litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCl2LitmusTest {
    pub name: String,
    pub description: String,
    pub work_items: usize,
    pub work_groups: usize,
    pub events: Vec<OpenCl2Event>,
    pub expected_outcomes: Vec<OpenCl2Outcome>,
    pub forbidden_outcomes: Vec<OpenCl2Outcome>,
}

/// An outcome of an OpenCL 2.0 litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenCl2Outcome {
    pub values: Vec<(String, u64)>,
}

impl OpenCl2Outcome {
    pub fn new(values: Vec<(String, u64)>) -> Self { Self { values } }
}

impl fmt::Display for OpenCl2Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.values.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl OpenCl2LitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(), description: String::new(),
            work_items: 2, work_groups: 1,
            events: Vec::new(),
            expected_outcomes: Vec::new(), forbidden_outcomes: Vec::new(),
        }
    }

    pub fn is_forbidden(&self, outcome: &OpenCl2Outcome) -> bool {
        self.forbidden_outcomes.contains(outcome)
    }

    /// Store Buffer test with device-scope acquire/release.
    pub fn store_buffer_device_scope() -> Self {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 1, 0, 0, 0);
        let mut test = Self::new("SB-OCL2");
        test.description = "Store Buffer at device scope".to_string();
        test.work_items = 2;

        let e0 = OpenCl2Event::new(0, wi0.clone(), OpenCl2OpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(OpenCl2MemoryOrder::Release).with_scope(OpenClScope::Device);
        let e1 = OpenCl2Event::new(1, wi0, OpenCl2OpType::AtomicLoad)
            .with_address(0x200)
            .with_order(OpenCl2MemoryOrder::Acquire).with_scope(OpenClScope::Device)
            .with_po(1);
        let e2 = OpenCl2Event::new(2, wi1.clone(), OpenCl2OpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_order(OpenCl2MemoryOrder::Release).with_scope(OpenClScope::Device);
        let e3 = OpenCl2Event::new(3, wi1, OpenCl2OpType::AtomicLoad)
            .with_address(0x100)
            .with_order(OpenCl2MemoryOrder::Acquire).with_scope(OpenClScope::Device)
            .with_po(1);

        test.events = vec![e0, e1, e2, e3];
        test.forbidden_outcomes.push(OpenCl2Outcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]));
        test
    }

    /// Message passing with work-group barrier in local memory.
    pub fn mp_local_barrier() -> Self {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 1, 0, 0, 0);
        let mut test = Self::new("MP-Local-OCL2");
        test.description = "Message passing with local memory and work-group barrier".to_string();
        test.work_items = 2;

        let e0 = OpenCl2Event::new(0, wi0.clone(), OpenCl2OpType::Store)
            .with_address(0x100).with_value(42).with_region(OpenClMemoryRegion::Local);
        let e1 = OpenCl2Event::new(1, wi0.clone(), OpenCl2OpType::WorkGroupBarrier)
            .with_scope(OpenClScope::WorkGroup).with_po(1);
        let e2 = OpenCl2Event::new(2, wi0, OpenCl2OpType::Store)
            .with_address(0x200).with_value(1).with_region(OpenClMemoryRegion::Local).with_po(2);
        let e3 = OpenCl2Event::new(3, wi1.clone(), OpenCl2OpType::Load)
            .with_address(0x200).with_region(OpenClMemoryRegion::Local);
        let e4 = OpenCl2Event::new(4, wi1.clone(), OpenCl2OpType::WorkGroupBarrier)
            .with_scope(OpenClScope::WorkGroup).with_po(1);
        let e5 = OpenCl2Event::new(5, wi1, OpenCl2OpType::Load)
            .with_address(0x100).with_region(OpenClMemoryRegion::Local).with_po(2);

        test.events = vec![e0, e1, e2, e3, e4, e5];
        test.forbidden_outcomes.push(OpenCl2Outcome::new(vec![
            ("flag".to_string(), 1), ("data".to_string(), 0),
        ]));
        test
    }

    /// SVM atomic test across devices.
    pub fn svm_cross_device() -> Self {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 0, 0, 0, 1);
        let mut test = Self::new("SVM-CrossDevice-OCL2");
        test.description = "SVM atomic operations across devices".to_string();
        test.work_items = 2;

        let e0 = OpenCl2Event::new(0, wi0.clone(), OpenCl2OpType::SvmAtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(OpenCl2MemoryOrder::Release).with_scope(OpenClScope::AllSvmDevices);
        let e1 = OpenCl2Event::new(1, wi1.clone(), OpenCl2OpType::SvmAtomicLoad)
            .with_address(0x100)
            .with_order(OpenCl2MemoryOrder::Acquire).with_scope(OpenClScope::AllSvmDevices);

        test.events = vec![e0, e1];
        test
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- OpenClScope tests --

    #[test]
    fn test_scope_ordering() {
        assert!(OpenClScope::AllSvmDevices > OpenClScope::Device);
        assert!(OpenClScope::Device > OpenClScope::WorkGroup);
        assert!(OpenClScope::WorkGroup > OpenClScope::SubGroup);
        assert!(OpenClScope::SubGroup > OpenClScope::WorkItem);
    }

    #[test]
    fn test_scope_includes() {
        assert!(OpenClScope::AllSvmDevices.includes(&OpenClScope::Device));
        assert!(OpenClScope::Device.includes(&OpenClScope::WorkGroup));
        assert!(!OpenClScope::WorkItem.includes(&OpenClScope::SubGroup));
    }

    #[test]
    fn test_scope_broaden_narrow() {
        assert_eq!(OpenClScope::WorkItem.broaden(), Some(OpenClScope::SubGroup));
        assert_eq!(OpenClScope::AllSvmDevices.broaden(), None);
        assert_eq!(OpenClScope::AllSvmDevices.narrow(), Some(OpenClScope::Device));
        assert_eq!(OpenClScope::WorkItem.narrow(), None);
    }

    #[test]
    fn test_scope_display() {
        assert_eq!(format!("{}", OpenClScope::WorkGroup), "work_group");
        assert_eq!(format!("{}", OpenClScope::AllSvmDevices), "all_svm_devices");
    }

    // -- OpenClMemoryRegion tests --

    #[test]
    fn test_memory_region() {
        assert!(OpenClMemoryRegion::Global.is_shared());
        assert!(OpenClMemoryRegion::Local.is_shared());
        assert!(!OpenClMemoryRegion::Private.is_shared());
        assert!(OpenClMemoryRegion::Local.is_work_group_local());
        assert!(!OpenClMemoryRegion::Global.is_work_group_local());
    }

    #[test]
    fn test_memory_region_min_scope() {
        assert_eq!(OpenClMemoryRegion::Private.min_sync_scope(), OpenClScope::WorkItem);
        assert_eq!(OpenClMemoryRegion::Local.min_sync_scope(), OpenClScope::WorkGroup);
        assert_eq!(OpenClMemoryRegion::Global.min_sync_scope(), OpenClScope::Device);
    }

    // -- OpenCl2MemoryOrder tests --

    #[test]
    fn test_memory_order() {
        assert!(OpenCl2MemoryOrder::Acquire.is_acquire());
        assert!(!OpenCl2MemoryOrder::Release.is_acquire());
        assert!(OpenCl2MemoryOrder::Release.is_release());
        assert!(OpenCl2MemoryOrder::AcqRel.is_acquire());
        assert!(OpenCl2MemoryOrder::AcqRel.is_release());
        assert!(OpenCl2MemoryOrder::SeqCst.is_acquire());
        assert!(OpenCl2MemoryOrder::SeqCst.is_release());
    }

    #[test]
    fn test_memory_order_at_least() {
        assert!(OpenCl2MemoryOrder::SeqCst.is_at_least(OpenCl2MemoryOrder::Relaxed));
        assert!(!OpenCl2MemoryOrder::Relaxed.is_at_least(OpenCl2MemoryOrder::Acquire));
    }

    #[test]
    fn test_memory_order_combine() {
        assert_eq!(
            OpenCl2MemoryOrder::Acquire.combine(OpenCl2MemoryOrder::Release),
            OpenCl2MemoryOrder::Release
        );
    }

    // -- WorkItemId tests --

    #[test]
    fn test_work_item_same_scope() {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 1, 0, 0, 0);
        let wi2 = WorkItemId::new(2, 0, 1, 1, 0);
        assert!(wi0.same_scope_instance(&wi1, OpenClScope::WorkGroup));
        assert!(!wi0.same_scope_instance(&wi2, OpenClScope::WorkGroup));
        assert!(wi0.same_scope_instance(&wi2, OpenClScope::Device));
    }

    // -- OpenCl2OpType tests --

    #[test]
    fn test_op_type() {
        assert!(OpenCl2OpType::Load.is_read());
        assert!(!OpenCl2OpType::Load.is_write());
        assert!(!OpenCl2OpType::Load.is_atomic());
        assert!(OpenCl2OpType::AtomicFetchAdd.is_rmw());
        assert!(OpenCl2OpType::WorkGroupBarrier.is_barrier());
        assert!(OpenCl2OpType::AtomicFence.is_fence());
        assert!(OpenCl2OpType::SvmAtomicLoad.is_svm());
        assert!(!OpenCl2OpType::AtomicLoad.is_svm());
    }

    #[test]
    fn test_barrier_scope() {
        assert_eq!(OpenCl2OpType::WorkGroupBarrier.barrier_scope(), Some(OpenClScope::WorkGroup));
        assert_eq!(OpenCl2OpType::SubGroupBarrier.barrier_scope(), Some(OpenClScope::SubGroup));
        assert_eq!(OpenCl2OpType::AtomicLoad.barrier_scope(), None);
    }

    // -- OpenCl2Event tests --

    #[test]
    fn test_event_creation() {
        let wi = WorkItemId::new(0, 0, 0, 0, 0);
        let event = OpenCl2Event::new(0, wi, OpenCl2OpType::AtomicStore)
            .with_address(0x100).with_value(42).with_scope(OpenClScope::Device);
        assert_eq!(event.id, 0);
        assert_eq!(event.address, 0x100);
        assert_eq!(event.value, 42);
    }

    #[test]
    fn test_event_display() {
        let wi = WorkItemId::new(0, 0, 0, 0, 0);
        let event = OpenCl2Event::new(0, wi, OpenCl2OpType::Store).with_address(0x100);
        let s = format!("{}", event);
        assert!(s.contains("E0"));
    }

    // -- SvmAtomicOp tests --

    #[test]
    fn test_svm_op_fine_grained() {
        let op = SvmAtomicOp::new(0, 0x100, OpenCl2OpType::SvmAtomicStore);
        assert!(op.requires_all_svm_scope());
        assert!(op.is_scope_valid()); // default is AllSvmDevices
    }

    #[test]
    fn test_svm_op_coarse_grained() {
        let mut op = SvmAtomicOp::new(0, 0x100, OpenCl2OpType::SvmAtomicStore);
        op.coarse_grained = true;
        op.scope = OpenClScope::Device;
        assert!(!op.requires_all_svm_scope());
        assert!(op.is_scope_valid());
    }

    #[test]
    fn test_svm_op_invalid_scope() {
        let mut op = SvmAtomicOp::new(0, 0x100, OpenCl2OpType::SvmAtomicStore);
        op.scope = OpenClScope::Device; // fine-grained needs AllSvmDevices
        assert!(!op.is_scope_valid());
    }

    // -- OrderingGraph tests --

    #[test]
    fn test_ordering_graph_no_cycle() {
        let mut g = OpenCl2OrderingGraph::new();
        g.add_edge(0, 1, OpenCl2Relation::ProgramOrder);
        g.add_edge(1, 2, OpenCl2Relation::ProgramOrder);
        assert!(!g.has_cycle());
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_ordering_graph_cycle() {
        let mut g = OpenCl2OrderingGraph::new();
        g.add_edge(0, 1, OpenCl2Relation::ProgramOrder);
        g.add_edge(1, 2, OpenCl2Relation::SynchronizesWith);
        g.add_edge(2, 0, OpenCl2Relation::ReadsFrom);
        assert!(g.has_cycle());
    }

    // -- OpenCl2Execution tests --

    #[test]
    fn test_execution_basic() {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 1, 0, 0, 0);
        let mut exec = OpenCl2Execution::new();
        exec.add_event(OpenCl2Event::new(0, wi0.clone(), OpenCl2OpType::Store));
        exec.add_event(OpenCl2Event::new(1, wi0, OpenCl2OpType::Load).with_po(1));
        exec.add_event(OpenCl2Event::new(2, wi1, OpenCl2OpType::Store));
        assert_eq!(exec.event_count(), 3);
        assert_eq!(exec.work_item_count(), 2);
    }

    #[test]
    fn test_execution_build_po() {
        let wi = WorkItemId::new(0, 0, 0, 0, 0);
        let mut exec = OpenCl2Execution::new();
        exec.add_event(OpenCl2Event::new(0, wi.clone(), OpenCl2OpType::Store).with_po(0));
        exec.add_event(OpenCl2Event::new(1, wi, OpenCl2OpType::Load).with_po(1));
        exec.build_program_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    #[test]
    fn test_execution_consistent() {
        let exec = OpenCl2Execution::new();
        assert!(exec.is_consistent());
    }

    // -- OpenCl2AxiomChecker tests --

    #[test]
    fn test_checker_empty_execution() {
        let exec = OpenCl2Execution::new();
        let checker = OpenCl2AxiomChecker::new(&exec);
        assert!(checker.check_all().is_empty());
    }

    #[test]
    fn test_checker_local_memory_violation() {
        let wi0 = WorkItemId::new(0, 0, 0, 0, 0);
        let wi1 = WorkItemId::new(1, 0, 0, 1, 0); // different work-group
        let mut exec = OpenCl2Execution::new();
        exec.add_event(OpenCl2Event::new(0, wi0, OpenCl2OpType::Store)
            .with_address(0x100).with_region(OpenClMemoryRegion::Local));
        exec.add_event(OpenCl2Event::new(1, wi1, OpenCl2OpType::Load)
            .with_address(0x100).with_region(OpenClMemoryRegion::Local));
        let checker = OpenCl2AxiomChecker::new(&exec);
        let violations = checker.check_local_memory_restriction();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_checker_svm_validity() {
        let mut exec = OpenCl2Execution::new();
        let mut op = SvmAtomicOp::new(0, 0x100, OpenCl2OpType::SvmAtomicStore);
        op.scope = OpenClScope::Device; // invalid for fine-grained
        exec.add_svm_op(op);
        let checker = OpenCl2AxiomChecker::new(&exec);
        let violations = checker.check_svm_scope_validity();
        assert!(!violations.is_empty());
    }

    // -- OpenCl2Model tests --

    #[test]
    fn test_model_verify_empty() {
        let model = OpenCl2Model::new();
        let exec = OpenCl2Execution::new();
        let result = model.verify(&exec);
        assert!(result.consistent);
        assert_eq!(result.model_name, "OpenCL2");
    }

    #[test]
    fn test_model_config() {
        let config = OpenCl2ModelConfig::full();
        assert!(config.check_coherence);
        assert!(config.check_svm);
        let config = OpenCl2ModelConfig::minimal();
        assert!(config.check_coherence);
        assert!(!config.check_svm);
    }

    // -- Litmus tests --

    #[test]
    fn test_litmus_store_buffer() {
        let test = OpenCl2LitmusTest::store_buffer_device_scope();
        assert_eq!(test.name, "SB-OCL2");
        assert_eq!(test.events.len(), 4);
        assert_eq!(test.forbidden_outcomes.len(), 1);
    }

    #[test]
    fn test_litmus_mp_local_barrier() {
        let test = OpenCl2LitmusTest::mp_local_barrier();
        assert_eq!(test.name, "MP-Local-OCL2");
        assert_eq!(test.events.len(), 6);
    }

    #[test]
    fn test_litmus_svm_cross_device() {
        let test = OpenCl2LitmusTest::svm_cross_device();
        assert_eq!(test.name, "SVM-CrossDevice-OCL2");
        assert_eq!(test.events.len(), 2);
    }

    #[test]
    fn test_litmus_forbidden() {
        let test = OpenCl2LitmusTest::store_buffer_device_scope();
        let outcome = OpenCl2Outcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]);
        assert!(test.is_forbidden(&outcome));
    }

    // -- Display tests --

    #[test]
    fn test_displays() {
        assert_eq!(format!("{}", OpenClScope::Device), "device");
        assert_eq!(format!("{}", OpenClMemoryRegion::Global), "global");
        assert_eq!(format!("{}", OpenCl2MemoryOrder::SeqCst), "seq_cst");
        assert_eq!(format!("{}", OpenCl2Relation::ProgramOrder), "po");
    }

    #[test]
    fn test_verification_result_display() {
        let result = OpenCl2VerificationResult {
            model_name: "OpenCL2".to_string(),
            consistent: true,
            violations: vec![],
            events_checked: 4,
            work_items: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("Consistent"));
    }

    #[test]
    fn test_outcome_display() {
        let o = OpenCl2Outcome::new(vec![("r0".to_string(), 1)]);
        assert!(format!("{}", o).contains("r0=1"));
    }

    #[test]
    fn test_axiom_all() {
        assert_eq!(OpenCl2Axiom::all().len(), 8);
    }

    #[test]
    fn test_axiom_display() {
        assert_eq!(format!("{}", OpenCl2Axiom::CoherencePerScope), "coherence-per-scope");
        assert_eq!(format!("{}", OpenCl2Axiom::SvmScopeValidity), "svm-scope-validity");
    }

    #[test]
    fn test_violation_display() {
        let v = OpenCl2Violation::new(OpenCl2Axiom::CoherencePerScope, "test", vec![0]);
        assert!(format!("{}", v).contains("coherence-per-scope"));
    }

    #[test]
    fn test_relation_display() {
        assert_eq!(format!("{}", OpenCl2Relation::SvmSync), "svm-sync");
        assert_eq!(format!("{}", OpenCl2Relation::BarrierOrder), "bar");
    }
}
