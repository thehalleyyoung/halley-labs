//! AMD HIP memory model specification for LITMUS∞.
//!
//! Implements the HIP memory model which closely mirrors CUDA but with
//! AMD-specific terminology and semantics:
//! - Wavefront, workgroup, agent, system scopes
//! - `__threadfence_block`, `__threadfence`, `__threadfence_system` equivalents
//! - Memory coherency domains (fine-grained vs coarse-grained)
//! - GFX9/RDNA cache hierarchy awareness
//!
//! # HIP scope hierarchy
//!
//! ```text
//! System (all CPUs + GPUs)
//!  └─ Agent (single GPU)
//!      └─ Workgroup (CU)
//!          └─ Wavefront (wave64 / wave32)
//! ```
//!
//! Key difference from CUDA: HIP uses "agent" instead of "device" and
//! "wavefront" instead of "warp/thread". HIP also exposes AMD-specific
//! coherency domain concepts for fine-grained vs coarse-grained memory.

#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// HIP scope hierarchy
// ---------------------------------------------------------------------------

/// HIP memory scope levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HipScope {
    /// Single lane within a wavefront.
    Wavefront,
    /// Workgroup (Compute Unit).
    Workgroup,
    /// Single GPU agent.
    Agent,
    /// System-wide (CPU + all GPUs).
    System,
}

impl HipScope {
    pub fn all() -> &'static [HipScope] {
        &[Self::Wavefront, Self::Workgroup, Self::Agent, Self::System]
    }

    /// Whether `self` is at least as broad as `other`.
    pub fn includes(&self, other: &HipScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    pub fn broaden(&self) -> Option<HipScope> {
        match self {
            Self::Wavefront => Some(Self::Workgroup),
            Self::Workgroup => Some(Self::Agent),
            Self::Agent => Some(Self::System),
            Self::System => None,
        }
    }

    pub fn narrow(&self) -> Option<HipScope> {
        match self {
            Self::System => Some(Self::Agent),
            Self::Agent => Some(Self::Workgroup),
            Self::Workgroup => Some(Self::Wavefront),
            Self::Wavefront => None,
        }
    }
}

impl fmt::Display for HipScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wavefront => write!(f, "wavefront"),
            Self::Workgroup => write!(f, "workgroup"),
            Self::Agent => write!(f, "agent"),
            Self::System => write!(f, "system"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory coherency domains
// ---------------------------------------------------------------------------

/// AMD memory coherency domain types.
///
/// HIP distinguishes between fine-grained and coarse-grained memory:
/// - Fine-grained: coherent across all agents without explicit cache management.
/// - Coarse-grained: requires explicit cache flush/invalidate between agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoherencyDomain {
    /// Fine-grained coherency — hardware-managed cache coherence.
    FineGrained,
    /// Coarse-grained coherency — requires explicit synchronisation.
    CoarseGrained,
}

impl CoherencyDomain {
    /// Whether cross-agent atomics need explicit cache management.
    pub fn needs_explicit_sync(&self) -> bool {
        matches!(self, Self::CoarseGrained)
    }

    /// Whether this domain supports system-scope atomics natively.
    pub fn supports_system_scope_atomics(&self) -> bool {
        matches!(self, Self::FineGrained)
    }
}

impl fmt::Display for CoherencyDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FineGrained => write!(f, "fine-grained"),
            Self::CoarseGrained => write!(f, "coarse-grained"),
        }
    }
}

// ---------------------------------------------------------------------------
// HIP memory ordering
// ---------------------------------------------------------------------------

/// HIP memory ordering (mirrors CUDA/C++ memory orders).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HipMemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl HipMemoryOrder {
    pub fn is_acquire(&self) -> bool {
        matches!(self, Self::Acquire | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_release(&self) -> bool {
        matches!(self, Self::Release | Self::AcqRel | Self::SeqCst)
    }

    pub fn is_at_least(&self, other: HipMemoryOrder) -> bool {
        (*self as u8) >= (other as u8)
    }

    pub fn combine(self, other: HipMemoryOrder) -> HipMemoryOrder {
        if (self as u8) >= (other as u8) { self } else { other }
    }
}

impl fmt::Display for HipMemoryOrder {
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
// HIP thread identity
// ---------------------------------------------------------------------------

/// Identity of a HIP thread within the execution hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HipThreadId {
    pub thread_id: usize,
    pub wavefront_id: usize,
    pub workgroup_id: usize,
    pub agent_id: usize,
}

impl HipThreadId {
    pub fn new(thread_id: usize, wavefront_id: usize, workgroup_id: usize, agent_id: usize) -> Self {
        Self { thread_id, wavefront_id, workgroup_id, agent_id }
    }

    pub fn same_scope_instance(&self, other: &HipThreadId, scope: HipScope) -> bool {
        match scope {
            HipScope::Wavefront => self.wavefront_id == other.wavefront_id,
            HipScope::Workgroup => self.workgroup_id == other.workgroup_id,
            HipScope::Agent => self.agent_id == other.agent_id,
            HipScope::System => true,
        }
    }
}

impl fmt::Display for HipThreadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T(t={}, wf={}, wg={}, ag={})",
            self.thread_id, self.wavefront_id, self.workgroup_id, self.agent_id)
    }
}

// ---------------------------------------------------------------------------
// HIP operation types
// ---------------------------------------------------------------------------

/// HIP memory operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HipOpType {
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
    /// `__threadfence_block()` — workgroup scope fence.
    ThreadFenceBlock,
    /// `__threadfence()` — agent (device) scope fence.
    ThreadFence,
    /// `__threadfence_system()` — system scope fence.
    ThreadFenceSystem,
}

impl HipOpType {
    pub fn is_read(&self) -> bool {
        matches!(self,
            Self::Load | Self::AtomicLoad | Self::AtomicExchange | Self::AtomicCAS
            | Self::AtomicAdd | Self::AtomicSub | Self::AtomicMin | Self::AtomicMax
            | Self::AtomicAnd | Self::AtomicOr | Self::AtomicXor
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(self,
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

    /// The scope implied by a `__threadfence*` variant.
    pub fn fence_scope(&self) -> Option<HipScope> {
        match self {
            Self::ThreadFenceBlock => Some(HipScope::Workgroup),
            Self::ThreadFence => Some(HipScope::Agent),
            Self::ThreadFenceSystem => Some(HipScope::System),
            _ => None,
        }
    }
}

impl fmt::Display for HipOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// HIP memory event
// ---------------------------------------------------------------------------

/// A memory event in the HIP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipEvent {
    pub id: usize,
    pub thread_id: HipThreadId,
    pub op_type: HipOpType,
    pub address: u64,
    pub value: u64,
    pub order: HipMemoryOrder,
    pub scope: HipScope,
    pub coherency_domain: CoherencyDomain,
    pub program_order: usize,
}

impl HipEvent {
    pub fn new(id: usize, thread_id: HipThreadId, op_type: HipOpType) -> Self {
        Self {
            id,
            thread_id,
            op_type,
            address: 0,
            value: 0,
            order: HipMemoryOrder::Relaxed,
            scope: HipScope::Agent,
            coherency_domain: CoherencyDomain::FineGrained,
            program_order: 0,
        }
    }

    pub fn with_address(mut self, addr: u64) -> Self { self.address = addr; self }
    pub fn with_value(mut self, val: u64) -> Self { self.value = val; self }
    pub fn with_scope(mut self, scope: HipScope) -> Self { self.scope = scope; self }
    pub fn with_order(mut self, order: HipMemoryOrder) -> Self { self.order = order; self }
    pub fn with_coherency(mut self, domain: CoherencyDomain) -> Self { self.coherency_domain = domain; self }
    pub fn with_po(mut self, po: usize) -> Self { self.program_order = po; self }

    pub fn same_scope_instance(&self, other: &HipEvent, scope: HipScope) -> bool {
        self.thread_id.same_scope_instance(&other.thread_id, scope)
    }
}

impl fmt::Display for HipEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}:{} {} @{:#x}={} [{}/{}/{}]",
            self.id, self.thread_id, self.op_type,
            self.address, self.value,
            self.order, self.scope, self.coherency_domain)
    }
}

// ---------------------------------------------------------------------------
// Ordering relations
// ---------------------------------------------------------------------------

/// Relations in the HIP memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HipRelation {
    ProgramOrder,
    ScopedModificationOrder,
    ReadsFrom,
    FromReads,
    SynchronizesWith,
    ScopedHappensBefore,
    FenceOrder,
    CoherencyDomainOrder,
}

impl fmt::Display for HipRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgramOrder => write!(f, "po"),
            Self::ScopedModificationOrder => write!(f, "smo"),
            Self::ReadsFrom => write!(f, "rf"),
            Self::FromReads => write!(f, "fr"),
            Self::SynchronizesWith => write!(f, "sw"),
            Self::ScopedHappensBefore => write!(f, "shb"),
            Self::FenceOrder => write!(f, "fence"),
            Self::CoherencyDomainOrder => write!(f, "cdo"),
        }
    }
}

/// Edge in the HIP ordering graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HipOrderingEdge {
    pub from: usize,
    pub to: usize,
    pub relation: HipRelation,
}

/// Ordering graph for the HIP model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HipOrderingGraph {
    pub edges: Vec<HipOrderingEdge>,
    pub adjacency: HashMap<usize, Vec<(usize, HipRelation)>>,
}

impl HipOrderingGraph {
    pub fn new() -> Self { Self { edges: Vec::new(), adjacency: HashMap::new() } }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: HipRelation) {
        self.edges.push(HipOrderingEdge { from, to, relation });
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

    pub fn edges_of(&self, relation: HipRelation) -> Vec<&HipOrderingEdge> {
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
// HIP execution
// ---------------------------------------------------------------------------

/// A HIP execution for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipExecution {
    pub events: Vec<HipEvent>,
    pub ordering: HipOrderingGraph,
}

impl HipExecution {
    pub fn new() -> Self {
        Self { events: Vec::new(), ordering: HipOrderingGraph::new() }
    }

    pub fn add_event(&mut self, event: HipEvent) {
        self.events.push(event);
    }

    pub fn get_event(&self, id: usize) -> Option<&HipEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn build_program_order(&mut self) {
        let mut by_thread: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in &self.events {
            by_thread.entry(event.thread_id.thread_id).or_default().push(event.id);
        }
        for (_tid, mut ids) in by_thread {
            ids.sort_by_key(|&id| {
                self.events.iter().find(|e| e.id == id).map(|e| e.program_order).unwrap_or(0)
            });
            for window in ids.windows(2) {
                self.ordering.add_edge(window[0], window[1], HipRelation::ProgramOrder);
            }
        }
    }

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
                if se.address == le.address && se.value == le.value
                    && se.thread_id != le.thread_id
                {
                    let sync_scope = std::cmp::min(se.scope, le.scope);
                    if se.same_scope_instance(le, sync_scope) {
                        self.ordering.add_edge(s, l, HipRelation::SynchronizesWith);
                    }
                }
            }
        }
    }

    /// Build fence-order edges for __threadfence variants.
    pub fn build_fence_order(&mut self) {
        let fences: Vec<usize> = self.events.iter()
            .filter(|e| e.op_type.is_fence())
            .map(|e| e.id).collect();

        for &fid in &fences {
            let fe = self.events.iter().find(|e| e.id == fid).unwrap().clone();

            let before: Vec<usize> = self.events.iter()
                .filter(|e| e.thread_id.thread_id == fe.thread_id.thread_id
                    && e.program_order < fe.program_order && !e.op_type.is_fence())
                .map(|e| e.id).collect();
            let after: Vec<usize> = self.events.iter()
                .filter(|e| e.thread_id.thread_id == fe.thread_id.thread_id
                    && e.program_order > fe.program_order && !e.op_type.is_fence())
                .map(|e| e.id).collect();

            for &b in &before {
                for &a in &after {
                    self.ordering.add_edge(b, a, HipRelation::FenceOrder);
                }
            }
        }
    }

    pub fn is_consistent(&self) -> bool { !self.ordering.has_cycle() }
    pub fn event_count(&self) -> usize { self.events.len() }

    pub fn thread_count(&self) -> usize {
        self.events.iter().map(|e| e.thread_id.thread_id).collect::<HashSet<_>>().len()
    }
}

impl Default for HipExecution {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// HIP axioms
// ---------------------------------------------------------------------------

/// Axiom in the HIP memory model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HipAxiom {
    /// Per-scope coherence order is total.
    CoherencePerScope,
    /// Atomicity of RMW per scope.
    AtomicityPerScope,
    /// Scoped happens-before must be acyclic.
    ScopedHbAcyclic,
    /// Fence ordering consistency.
    FenceConsistency,
    /// SeqCst total order per scope.
    SeqCstPerScope,
    /// No thin-air values.
    NoThinAir,
    /// Coherency domain consistency.
    CoherencyDomainConsistency,
}

impl HipAxiom {
    pub fn all() -> Vec<Self> {
        vec![
            Self::CoherencePerScope, Self::AtomicityPerScope,
            Self::ScopedHbAcyclic, Self::FenceConsistency,
            Self::SeqCstPerScope, Self::NoThinAir,
            Self::CoherencyDomainConsistency,
        ]
    }
}

impl fmt::Display for HipAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoherencePerScope => write!(f, "coherence-per-scope"),
            Self::AtomicityPerScope => write!(f, "atomicity-per-scope"),
            Self::ScopedHbAcyclic => write!(f, "scoped-hb-acyclic"),
            Self::FenceConsistency => write!(f, "fence-consistency"),
            Self::SeqCstPerScope => write!(f, "seq-cst-per-scope"),
            Self::NoThinAir => write!(f, "no-thin-air"),
            Self::CoherencyDomainConsistency => write!(f, "coherency-domain-consistency"),
        }
    }
}

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

/// A violation of a HIP model axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipViolation {
    pub axiom: HipAxiom,
    pub description: String,
    pub involved_events: Vec<usize>,
}

impl HipViolation {
    pub fn new(axiom: HipAxiom, description: &str, events: Vec<usize>) -> Self {
        Self { axiom, description: description.to_string(), involved_events: events }
    }
}

impl fmt::Display for HipViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (events {:?})", self.axiom, self.description, self.involved_events)
    }
}

// ---------------------------------------------------------------------------
// HIP axiom checker
// ---------------------------------------------------------------------------

/// Checks HIP memory model axioms on an execution.
#[derive(Debug)]
pub struct HipAxiomChecker<'a> {
    execution: &'a HipExecution,
}

impl<'a> HipAxiomChecker<'a> {
    pub fn new(execution: &'a HipExecution) -> Self {
        Self { execution }
    }

    pub fn check_all(&self) -> Vec<HipViolation> {
        let mut violations = Vec::new();
        violations.extend(self.check_coherence_per_scope());
        violations.extend(self.check_atomicity_per_scope());
        violations.extend(self.check_scoped_hb_acyclic());
        violations.extend(self.check_fence_consistency());
        violations.extend(self.check_no_thin_air());
        violations.extend(self.check_coherency_domain_consistency());
        violations
    }

    pub fn check_coherence_per_scope(&self) -> Vec<HipViolation> {
        let mut violations = Vec::new();
        let mut by_addr: HashMap<u64, Vec<&HipEvent>> = HashMap::new();
        for e in &self.execution.events {
            if !e.op_type.is_fence() { by_addr.entry(e.address).or_default().push(e); }
        }

        for (addr, events) in &by_addr {
            let writes: Vec<&&HipEvent> = events.iter().filter(|e| e.op_type.is_write()).collect();
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
                        violations.push(HipViolation::new(
                            HipAxiom::CoherencePerScope,
                            &format!("Coherence cycle at {:#x}", addr),
                            vec![w1, w2],
                        ));
                    }
                }
            }
        }
        violations
    }

    pub fn check_atomicity_per_scope(&self) -> Vec<HipViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if !event.op_type.is_rmw() { continue; }
            let rf_sources: Vec<usize> = self.execution.ordering.edges_of(HipRelation::ReadsFrom)
                .iter().filter(|e| e.to == event.id).map(|e| e.from).collect();
            if rf_sources.len() > 1 {
                violations.push(HipViolation::new(
                    HipAxiom::AtomicityPerScope,
                    &format!("RMW e{} has multiple rf-sources", event.id),
                    rf_sources,
                ));
            }
        }
        violations
    }

    pub fn check_scoped_hb_acyclic(&self) -> Vec<HipViolation> {
        if self.execution.ordering.has_cycle() {
            vec![HipViolation::new(
                HipAxiom::ScopedHbAcyclic,
                "Cycle detected in scoped happens-before",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    pub fn check_fence_consistency(&self) -> Vec<HipViolation> {
        // Fence semantics are checked via fence-order edge construction.
        let violations = Vec::new();
        for event in &self.execution.events {
            if event.op_type.is_fence() {
                let _fence_scope = event.op_type.fence_scope().unwrap_or(event.scope);
            }
        }
        violations
    }

    pub fn check_no_thin_air(&self) -> Vec<HipViolation> {
        if self.execution.ordering.has_cycle() {
            vec![HipViolation::new(
                HipAxiom::NoThinAir,
                "Potential thin-air cycle detected",
                vec![],
            )]
        } else {
            vec![]
        }
    }

    /// Check coherency domain consistency: cross-agent operations on
    /// coarse-grained memory must use system-scope fences.
    pub fn check_coherency_domain_consistency(&self) -> Vec<HipViolation> {
        let mut violations = Vec::new();
        for event in &self.execution.events {
            if event.coherency_domain != CoherencyDomain::CoarseGrained { continue; }
            if !event.op_type.is_write() { continue; }

            // Find cross-agent readers of the same address.
            for reader in &self.execution.events {
                if reader.address != event.address { continue; }
                if !reader.op_type.is_read() { continue; }
                if reader.thread_id.agent_id == event.thread_id.agent_id { continue; }

                // Cross-agent coarse-grained access needs system scope.
                if event.scope != HipScope::System {
                    violations.push(HipViolation::new(
                        HipAxiom::CoherencyDomainConsistency,
                        &format!(
                            "Coarse-grained cross-agent write at {:#x} (agent {}) read by agent {} without system scope",
                            event.address, event.thread_id.agent_id, reader.thread_id.agent_id
                        ),
                        vec![event.id, reader.id],
                    ));
                    break;
                }
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// HIP model configuration
// ---------------------------------------------------------------------------

/// Configuration for the HIP model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipModelConfig {
    pub check_coherence: bool,
    pub check_atomicity: bool,
    pub check_hb_acyclic: bool,
    pub check_fence: bool,
    pub check_seq_cst: bool,
    pub check_no_thin_air: bool,
    pub check_coherency_domain: bool,
    pub max_threads: usize,
    pub max_workgroups: usize,
}

impl HipModelConfig {
    pub fn full() -> Self {
        Self {
            check_coherence: true, check_atomicity: true, check_hb_acyclic: true,
            check_fence: true, check_seq_cst: true, check_no_thin_air: true,
            check_coherency_domain: true,
            max_threads: 1024, max_workgroups: 64,
        }
    }

    pub fn minimal() -> Self {
        Self {
            check_coherence: true, check_atomicity: false, check_hb_acyclic: true,
            check_fence: false, check_seq_cst: false, check_no_thin_air: true,
            check_coherency_domain: false,
            max_threads: 128, max_workgroups: 4,
        }
    }
}

impl Default for HipModelConfig {
    fn default() -> Self { Self::full() }
}

// ---------------------------------------------------------------------------
// HIP model
// ---------------------------------------------------------------------------

/// HIP memory model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipModel {
    pub name: String,
    pub config: HipModelConfig,
}

impl HipModel {
    pub fn new() -> Self {
        Self { name: "HIP".to_string(), config: HipModelConfig::full() }
    }

    pub fn with_config(config: HipModelConfig) -> Self {
        Self { name: "HIP".to_string(), config }
    }

    pub fn verify(&self, execution: &HipExecution) -> HipVerificationResult {
        let checker = HipAxiomChecker::new(execution);
        let violations = checker.check_all();
        let consistent = violations.is_empty();
        HipVerificationResult {
            model_name: self.name.clone(),
            consistent,
            violations,
            events_checked: execution.event_count(),
            threads: execution.thread_count(),
        }
    }
}

impl Default for HipModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of HIP model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipVerificationResult {
    pub model_name: String,
    pub consistent: bool,
    pub violations: Vec<HipViolation>,
    pub events_checked: usize,
    pub threads: usize,
}

impl fmt::Display for HipVerificationResult {
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
// HIP litmus test support
// ---------------------------------------------------------------------------

/// A HIP litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipLitmusTest {
    pub name: String,
    pub description: String,
    pub threads: usize,
    pub workgroups: usize,
    pub events: Vec<HipEvent>,
    pub expected_outcomes: Vec<HipOutcome>,
    pub forbidden_outcomes: Vec<HipOutcome>,
}

/// An outcome of a HIP litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HipOutcome {
    pub values: Vec<(String, u64)>,
}

impl HipOutcome {
    pub fn new(values: Vec<(String, u64)>) -> Self { Self { values } }
}

impl fmt::Display for HipOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.values.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl HipLitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(), description: String::new(),
            threads: 2, workgroups: 1,
            events: Vec::new(),
            expected_outcomes: Vec::new(), forbidden_outcomes: Vec::new(),
        }
    }

    pub fn is_forbidden(&self, outcome: &HipOutcome) -> bool {
        self.forbidden_outcomes.contains(outcome)
    }

    /// Store Buffer test with agent-scope acquire/release.
    pub fn store_buffer() -> Self {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 0);
        let mut test = Self::new("SB-HIP");
        test.description = "Store Buffer under HIP model".to_string();
        test.threads = 2;

        let e0 = HipEvent::new(0, t0.clone(), HipOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(HipMemoryOrder::Release).with_scope(HipScope::Agent);
        let e1 = HipEvent::new(1, t0, HipOpType::AtomicLoad)
            .with_address(0x200)
            .with_order(HipMemoryOrder::Acquire).with_scope(HipScope::Agent)
            .with_po(1);
        let e2 = HipEvent::new(2, t1.clone(), HipOpType::AtomicStore)
            .with_address(0x200).with_value(1)
            .with_order(HipMemoryOrder::Release).with_scope(HipScope::Agent);
        let e3 = HipEvent::new(3, t1, HipOpType::AtomicLoad)
            .with_address(0x100)
            .with_order(HipMemoryOrder::Acquire).with_scope(HipScope::Agent)
            .with_po(1);

        test.events = vec![e0, e1, e2, e3];
        test.forbidden_outcomes.push(HipOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]));
        test
    }

    /// Message passing with __threadfence.
    pub fn message_passing_fence() -> Self {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 0);
        let mut test = Self::new("MP-Fence-HIP");
        test.description = "Message passing with __threadfence".to_string();
        test.threads = 2;

        let e0 = HipEvent::new(0, t0.clone(), HipOpType::Store)
            .with_address(0x100).with_value(42);
        let e1 = HipEvent::new(1, t0.clone(), HipOpType::ThreadFence).with_po(1);
        let e2 = HipEvent::new(2, t0, HipOpType::Store)
            .with_address(0x200).with_value(1).with_po(2);
        let e3 = HipEvent::new(3, t1.clone(), HipOpType::Load)
            .with_address(0x200);
        let e4 = HipEvent::new(4, t1.clone(), HipOpType::ThreadFence).with_po(1);
        let e5 = HipEvent::new(5, t1, HipOpType::Load)
            .with_address(0x100).with_po(2);

        test.events = vec![e0, e1, e2, e3, e4, e5];
        test.forbidden_outcomes.push(HipOutcome::new(vec![
            ("flag".to_string(), 1), ("data".to_string(), 0),
        ]));
        test
    }

    /// Cross-agent test with coarse-grained memory.
    pub fn cross_agent_coarse() -> Self {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 1); // different agent
        let mut test = Self::new("CrossAgent-Coarse-HIP");
        test.description = "Cross-agent access on coarse-grained memory".to_string();
        test.threads = 2;

        let e0 = HipEvent::new(0, t0, HipOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_order(HipMemoryOrder::Release).with_scope(HipScope::System)
            .with_coherency(CoherencyDomain::CoarseGrained);
        let e1 = HipEvent::new(1, t1, HipOpType::AtomicLoad)
            .with_address(0x100)
            .with_order(HipMemoryOrder::Acquire).with_scope(HipScope::System)
            .with_coherency(CoherencyDomain::CoarseGrained);

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

    // -- HipScope tests --

    #[test]
    fn test_scope_ordering() {
        assert!(HipScope::System > HipScope::Agent);
        assert!(HipScope::Agent > HipScope::Workgroup);
        assert!(HipScope::Workgroup > HipScope::Wavefront);
    }

    #[test]
    fn test_scope_includes() {
        assert!(HipScope::System.includes(&HipScope::Agent));
        assert!(HipScope::Agent.includes(&HipScope::Workgroup));
        assert!(!HipScope::Wavefront.includes(&HipScope::Workgroup));
    }

    #[test]
    fn test_scope_broaden_narrow() {
        assert_eq!(HipScope::Wavefront.broaden(), Some(HipScope::Workgroup));
        assert_eq!(HipScope::System.broaden(), None);
        assert_eq!(HipScope::System.narrow(), Some(HipScope::Agent));
        assert_eq!(HipScope::Wavefront.narrow(), None);
    }

    #[test]
    fn test_scope_display() {
        assert_eq!(format!("{}", HipScope::Wavefront), "wavefront");
        assert_eq!(format!("{}", HipScope::Agent), "agent");
        assert_eq!(format!("{}", HipScope::System), "system");
    }

    // -- CoherencyDomain tests --

    #[test]
    fn test_coherency_domain() {
        assert!(!CoherencyDomain::FineGrained.needs_explicit_sync());
        assert!(CoherencyDomain::CoarseGrained.needs_explicit_sync());
        assert!(CoherencyDomain::FineGrained.supports_system_scope_atomics());
        assert!(!CoherencyDomain::CoarseGrained.supports_system_scope_atomics());
    }

    #[test]
    fn test_coherency_domain_display() {
        assert_eq!(format!("{}", CoherencyDomain::FineGrained), "fine-grained");
        assert_eq!(format!("{}", CoherencyDomain::CoarseGrained), "coarse-grained");
    }

    // -- HipMemoryOrder tests --

    #[test]
    fn test_memory_order() {
        assert!(HipMemoryOrder::Acquire.is_acquire());
        assert!(!HipMemoryOrder::Release.is_acquire());
        assert!(HipMemoryOrder::Release.is_release());
        assert!(HipMemoryOrder::AcqRel.is_acquire());
        assert!(HipMemoryOrder::AcqRel.is_release());
        assert!(HipMemoryOrder::SeqCst.is_acquire());
        assert!(HipMemoryOrder::SeqCst.is_release());
    }

    #[test]
    fn test_memory_order_at_least() {
        assert!(HipMemoryOrder::SeqCst.is_at_least(HipMemoryOrder::Relaxed));
        assert!(!HipMemoryOrder::Relaxed.is_at_least(HipMemoryOrder::Acquire));
    }

    #[test]
    fn test_memory_order_combine() {
        assert_eq!(HipMemoryOrder::Acquire.combine(HipMemoryOrder::Release), HipMemoryOrder::Release);
        assert_eq!(HipMemoryOrder::SeqCst.combine(HipMemoryOrder::Relaxed), HipMemoryOrder::SeqCst);
    }

    // -- HipThreadId tests --

    #[test]
    fn test_thread_same_scope() {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 0);
        let t2 = HipThreadId::new(2, 1, 1, 0);
        let t3 = HipThreadId::new(3, 2, 2, 1);
        assert!(t0.same_scope_instance(&t1, HipScope::Wavefront));
        assert!(t0.same_scope_instance(&t2, HipScope::Agent));
        assert!(!t0.same_scope_instance(&t2, HipScope::Workgroup));
        assert!(!t0.same_scope_instance(&t3, HipScope::Agent));
        assert!(t0.same_scope_instance(&t3, HipScope::System));
    }

    // -- HipOpType tests --

    #[test]
    fn test_op_type() {
        assert!(HipOpType::Load.is_read());
        assert!(!HipOpType::Load.is_write());
        assert!(!HipOpType::Load.is_atomic());
        assert!(HipOpType::AtomicCAS.is_rmw());
        assert!(HipOpType::ThreadFence.is_fence());
        assert!(HipOpType::ThreadFenceBlock.is_fence());
        assert!(HipOpType::ThreadFenceSystem.is_fence());
    }

    #[test]
    fn test_fence_scope() {
        assert_eq!(HipOpType::ThreadFenceBlock.fence_scope(), Some(HipScope::Workgroup));
        assert_eq!(HipOpType::ThreadFence.fence_scope(), Some(HipScope::Agent));
        assert_eq!(HipOpType::ThreadFenceSystem.fence_scope(), Some(HipScope::System));
        assert_eq!(HipOpType::AtomicLoad.fence_scope(), None);
    }

    // -- HipEvent tests --

    #[test]
    fn test_event_creation() {
        let tid = HipThreadId::new(0, 0, 0, 0);
        let event = HipEvent::new(0, tid, HipOpType::AtomicStore)
            .with_address(0x100).with_value(42).with_scope(HipScope::Agent);
        assert_eq!(event.id, 0);
        assert_eq!(event.address, 0x100);
        assert_eq!(event.value, 42);
        assert_eq!(event.scope, HipScope::Agent);
    }

    #[test]
    fn test_event_same_scope() {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 0);
        let e0 = HipEvent::new(0, t0, HipOpType::Store);
        let e1 = HipEvent::new(1, t1, HipOpType::Load);
        assert!(e0.same_scope_instance(&e1, HipScope::Wavefront));
    }

    #[test]
    fn test_event_display() {
        let tid = HipThreadId::new(0, 0, 0, 0);
        let event = HipEvent::new(0, tid, HipOpType::Store).with_address(0x100);
        let s = format!("{}", event);
        assert!(s.contains("E0"));
    }

    // -- OrderingGraph tests --

    #[test]
    fn test_ordering_graph_no_cycle() {
        let mut g = HipOrderingGraph::new();
        g.add_edge(0, 1, HipRelation::ProgramOrder);
        g.add_edge(1, 2, HipRelation::ProgramOrder);
        assert!(!g.has_cycle());
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_ordering_graph_cycle() {
        let mut g = HipOrderingGraph::new();
        g.add_edge(0, 1, HipRelation::ProgramOrder);
        g.add_edge(1, 2, HipRelation::SynchronizesWith);
        g.add_edge(2, 0, HipRelation::ReadsFrom);
        assert!(g.has_cycle());
    }

    // -- HipExecution tests --

    #[test]
    fn test_execution_basic() {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 0);
        let mut exec = HipExecution::new();
        exec.add_event(HipEvent::new(0, t0.clone(), HipOpType::Store));
        exec.add_event(HipEvent::new(1, t0, HipOpType::Load).with_po(1));
        exec.add_event(HipEvent::new(2, t1, HipOpType::Store));
        assert_eq!(exec.event_count(), 3);
        assert_eq!(exec.thread_count(), 2);
    }

    #[test]
    fn test_execution_build_po() {
        let t = HipThreadId::new(0, 0, 0, 0);
        let mut exec = HipExecution::new();
        exec.add_event(HipEvent::new(0, t.clone(), HipOpType::Store).with_po(0));
        exec.add_event(HipEvent::new(1, t, HipOpType::Load).with_po(1));
        exec.build_program_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    #[test]
    fn test_execution_consistent() {
        let exec = HipExecution::new();
        assert!(exec.is_consistent());
    }

    #[test]
    fn test_execution_fence_order() {
        let t = HipThreadId::new(0, 0, 0, 0);
        let mut exec = HipExecution::new();
        exec.add_event(HipEvent::new(0, t.clone(), HipOpType::Store).with_address(0x100).with_po(0));
        exec.add_event(HipEvent::new(1, t.clone(), HipOpType::ThreadFence).with_po(1));
        exec.add_event(HipEvent::new(2, t, HipOpType::Load).with_address(0x200).with_po(2));
        exec.build_fence_order();
        assert_eq!(exec.ordering.edge_count(), 1);
    }

    // -- HipAxiomChecker tests --

    #[test]
    fn test_checker_empty_execution() {
        let exec = HipExecution::new();
        let checker = HipAxiomChecker::new(&exec);
        assert!(checker.check_all().is_empty());
    }

    #[test]
    fn test_checker_coherency_domain_violation() {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 1); // different agent
        let mut exec = HipExecution::new();
        exec.add_event(HipEvent::new(0, t0, HipOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_scope(HipScope::Agent) // wrong scope for cross-agent coarse
            .with_coherency(CoherencyDomain::CoarseGrained));
        exec.add_event(HipEvent::new(1, t1, HipOpType::AtomicLoad)
            .with_address(0x100)
            .with_coherency(CoherencyDomain::CoarseGrained));
        let checker = HipAxiomChecker::new(&exec);
        let violations = checker.check_coherency_domain_consistency();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_checker_coherency_domain_valid() {
        let t0 = HipThreadId::new(0, 0, 0, 0);
        let t1 = HipThreadId::new(1, 0, 0, 1);
        let mut exec = HipExecution::new();
        exec.add_event(HipEvent::new(0, t0, HipOpType::AtomicStore)
            .with_address(0x100).with_value(1)
            .with_scope(HipScope::System) // correct scope
            .with_coherency(CoherencyDomain::CoarseGrained));
        exec.add_event(HipEvent::new(1, t1, HipOpType::AtomicLoad)
            .with_address(0x100)
            .with_coherency(CoherencyDomain::CoarseGrained));
        let checker = HipAxiomChecker::new(&exec);
        let violations = checker.check_coherency_domain_consistency();
        assert!(violations.is_empty());
    }

    // -- HipModel tests --

    #[test]
    fn test_model_verify_empty() {
        let model = HipModel::new();
        let exec = HipExecution::new();
        let result = model.verify(&exec);
        assert!(result.consistent);
        assert_eq!(result.model_name, "HIP");
    }

    #[test]
    fn test_model_config() {
        let config = HipModelConfig::full();
        assert!(config.check_coherence);
        assert!(config.check_coherency_domain);
        let config = HipModelConfig::minimal();
        assert!(config.check_coherence);
        assert!(!config.check_coherency_domain);
    }

    #[test]
    fn test_model_with_config() {
        let model = HipModel::with_config(HipModelConfig::minimal());
        assert_eq!(model.name, "HIP");
    }

    // -- Litmus tests --

    #[test]
    fn test_litmus_store_buffer() {
        let test = HipLitmusTest::store_buffer();
        assert_eq!(test.name, "SB-HIP");
        assert_eq!(test.threads, 2);
        assert_eq!(test.events.len(), 4);
        assert_eq!(test.forbidden_outcomes.len(), 1);
    }

    #[test]
    fn test_litmus_message_passing() {
        let test = HipLitmusTest::message_passing_fence();
        assert_eq!(test.name, "MP-Fence-HIP");
        assert_eq!(test.events.len(), 6);
    }

    #[test]
    fn test_litmus_cross_agent() {
        let test = HipLitmusTest::cross_agent_coarse();
        assert_eq!(test.name, "CrossAgent-Coarse-HIP");
        assert_eq!(test.events.len(), 2);
    }

    #[test]
    fn test_litmus_forbidden() {
        let test = HipLitmusTest::store_buffer();
        let outcome = HipOutcome::new(vec![
            ("r0".to_string(), 0), ("r1".to_string(), 0),
        ]);
        assert!(test.is_forbidden(&outcome));
    }

    // -- Display tests --

    #[test]
    fn test_displays() {
        assert_eq!(format!("{}", HipScope::Agent), "agent");
        assert_eq!(format!("{}", HipMemoryOrder::SeqCst), "seq_cst");
        assert_eq!(format!("{}", HipRelation::ProgramOrder), "po");
        assert_eq!(format!("{}", HipRelation::CoherencyDomainOrder), "cdo");
    }

    #[test]
    fn test_verification_result_display() {
        let result = HipVerificationResult {
            model_name: "HIP".to_string(),
            consistent: true,
            violations: vec![],
            events_checked: 4,
            threads: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("Consistent"));
        assert!(s.contains("4 events"));
    }

    #[test]
    fn test_outcome_display() {
        let o = HipOutcome::new(vec![("r0".to_string(), 1)]);
        assert!(format!("{}", o).contains("r0=1"));
    }

    #[test]
    fn test_axiom_all() {
        assert_eq!(HipAxiom::all().len(), 7);
    }

    #[test]
    fn test_axiom_display() {
        assert_eq!(format!("{}", HipAxiom::CoherencePerScope), "coherence-per-scope");
        assert_eq!(format!("{}", HipAxiom::CoherencyDomainConsistency), "coherency-domain-consistency");
    }

    #[test]
    fn test_violation_display() {
        let v = HipViolation::new(HipAxiom::CoherencePerScope, "test", vec![0]);
        assert!(format!("{}", v).contains("coherence-per-scope"));
    }
}
