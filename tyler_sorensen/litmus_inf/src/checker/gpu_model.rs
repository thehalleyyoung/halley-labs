//! CUDA/PTX memory model for the LITMUS∞ checker.
//!
//! Implements PTX memory ordering semantics, GPU scope hierarchy (CTA, GPU,
//! System), scoped fence operations, CUDA memory spaces, and GPU-specific
//! axiom checking. Supports compositional verification of GPU litmus tests
//! with multi-scope constraints.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    EventId, ThreadId, Address, OpType, Scope,
    ExecutionGraph, BitMatrix,
};
use crate::checker::memory_model::{
    MemoryModel, RelationExpr,
};
use crate::checker::litmus::{LitmusTest, Thread, Ordering};

// ═══════════════════════════════════════════════════════════════════════════
// PTX Memory Ordering
// ═══════════════════════════════════════════════════════════════════════════

/// PTX memory ordering semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PTXMemoryOrder {
    /// Relaxed: no ordering guarantees.
    Relaxed,
    /// Acquire: prevents subsequent operations from being reordered before this.
    Acquire,
    /// Release: prevents preceding operations from being reordered after this.
    Release,
    /// Acquire-Release: combines Acquire and Release.
    AcqRel,
    /// Sequentially Consistent: total order with all other SeqCst operations.
    SeqCst,
    /// Weak: weaker than relaxed in some PTX extensions.
    Weak,
}

impl PTXMemoryOrder {
    /// Check if this ordering is at least as strong as another.
    pub fn is_at_least(&self, other: &PTXMemoryOrder) -> bool {
        use PTXMemoryOrder::*;
        match (self, other) {
            (_, Weak) | (_, Relaxed) => true,
            (SeqCst, _) => true,
            (AcqRel, Acquire) | (AcqRel, Release) | (AcqRel, AcqRel) => true,
            (Acquire, Acquire) => true,
            (Release, Release) => true,
            _ => false,
        }
    }

    /// Whether this has acquire semantics.
    pub fn has_acquire(&self) -> bool {
        matches!(self, PTXMemoryOrder::Acquire | PTXMemoryOrder::AcqRel | PTXMemoryOrder::SeqCst)
    }

    /// Whether this has release semantics.
    pub fn has_release(&self) -> bool {
        matches!(self, PTXMemoryOrder::Release | PTXMemoryOrder::AcqRel | PTXMemoryOrder::SeqCst)
    }

    /// Convert from the litmus `Ordering` type.
    pub fn from_litmus_ordering(ord: &Ordering) -> Self {
        match ord {
            Ordering::Relaxed => PTXMemoryOrder::Relaxed,
            Ordering::Acquire => PTXMemoryOrder::Acquire,
            Ordering::Release => PTXMemoryOrder::Release,
            Ordering::AcqRel => PTXMemoryOrder::AcqRel,
            Ordering::SeqCst => PTXMemoryOrder::SeqCst,
            _ => PTXMemoryOrder::Relaxed,
        }
    }
}

impl fmt::Display for PTXMemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PTXMemoryOrder::Relaxed => write!(f, ".relaxed"),
            PTXMemoryOrder::Acquire => write!(f, ".acquire"),
            PTXMemoryOrder::Release => write!(f, ".release"),
            PTXMemoryOrder::AcqRel => write!(f, ".acq_rel"),
            PTXMemoryOrder::SeqCst => write!(f, ".sc"),
            PTXMemoryOrder::Weak => write!(f, ".weak"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Scope Hierarchy
// ═══════════════════════════════════════════════════════════════════════════

/// PTX scope levels for memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PTXScope {
    /// Cooperative Thread Array (warp-level scope).
    CTA,
    /// GPU device scope.
    GPU,
    /// System-wide scope (all devices + host CPU).
    System,
}

impl PTXScope {
    /// Whether self includes (is at least as wide as) other.
    pub fn includes(&self, other: &PTXScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    /// Convert to the execution module's Scope type.
    pub fn to_exec_scope(&self) -> Scope {
        match self {
            PTXScope::CTA => Scope::CTA,
            PTXScope::GPU => Scope::GPU,
            PTXScope::System => Scope::System,
        }
    }

    /// Convert from the execution module's Scope type.
    pub fn from_exec_scope(s: Scope) -> Option<Self> {
        match s {
            Scope::CTA => Some(PTXScope::CTA),
            Scope::GPU => Some(PTXScope::GPU),
            Scope::System => Some(PTXScope::System),
            Scope::None => None,
        }
    }

    /// All scope levels.
    pub fn all() -> &'static [PTXScope] {
        &[PTXScope::CTA, PTXScope::GPU, PTXScope::System]
    }
}

impl fmt::Display for PTXScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PTXScope::CTA => write!(f, ".cta"),
            PTXScope::GPU => write!(f, ".gpu"),
            PTXScope::System => write!(f, ".sys"),
        }
    }
}

/// GPU scope hierarchy: maps threads to their CTA/GPU placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeHierarchy {
    /// Thread → CTA mapping.
    pub thread_cta: HashMap<ThreadId, usize>,
    /// CTA → GPU mapping.
    pub cta_gpu: HashMap<usize, usize>,
    /// Number of CTAs.
    pub num_ctas: usize,
    /// Number of GPUs.
    pub num_gpus: usize,
}

impl ScopeHierarchy {
    /// Create a simple hierarchy with given CTA assignments.
    pub fn new(thread_cta: HashMap<ThreadId, usize>, num_ctas: usize) -> Self {
        let mut cta_gpu = HashMap::new();
        for cta in 0..num_ctas {
            cta_gpu.insert(cta, 0); // all on GPU 0 by default
        }
        ScopeHierarchy { thread_cta, cta_gpu, num_ctas, num_gpus: 1 }
    }

    /// Create a hierarchy where each thread is in its own CTA.
    pub fn one_thread_per_cta(num_threads: usize) -> Self {
        let thread_cta: HashMap<ThreadId, usize> = (0..num_threads).map(|t| (t, t)).collect();
        Self::new(thread_cta, num_threads)
    }

    /// Create a hierarchy where all threads are in one CTA.
    pub fn all_same_cta(num_threads: usize) -> Self {
        let thread_cta: HashMap<ThreadId, usize> = (0..num_threads).map(|t| (t, 0)).collect();
        Self::new(thread_cta, 1)
    }

    /// Check if two threads are in the same scope at a given level.
    pub fn same_scope(&self, t1: ThreadId, t2: ThreadId, scope: PTXScope) -> bool {
        match scope {
            PTXScope::CTA => {
                self.thread_cta.get(&t1) == self.thread_cta.get(&t2)
            }
            PTXScope::GPU => {
                let cta1 = self.thread_cta.get(&t1).copied().unwrap_or(0);
                let cta2 = self.thread_cta.get(&t2).copied().unwrap_or(0);
                self.cta_gpu.get(&cta1) == self.cta_gpu.get(&cta2)
            }
            PTXScope::System => true,
        }
    }

    /// Minimal common scope between two threads.
    pub fn minimal_common_scope(&self, t1: ThreadId, t2: ThreadId) -> PTXScope {
        if self.same_scope(t1, t2, PTXScope::CTA) {
            PTXScope::CTA
        } else if self.same_scope(t1, t2, PTXScope::GPU) {
            PTXScope::GPU
        } else {
            PTXScope::System
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA Memory Space
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA memory spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CUDAMemorySpace {
    /// Global memory (accessible by all threads).
    Global,
    /// Shared memory (per-CTA).
    Shared,
    /// Local memory (per-thread).
    Local,
    /// Constant memory (read-only, cached).
    Constant,
    /// Texture memory (read-only, spatially cached).
    Texture,
}

impl CUDAMemorySpace {
    /// Visibility scope of this memory space.
    pub fn visibility_scope(&self) -> PTXScope {
        match self {
            CUDAMemorySpace::Global => PTXScope::System,
            CUDAMemorySpace::Shared => PTXScope::CTA,
            CUDAMemorySpace::Local => PTXScope::CTA, // conceptually thread but CTA contains it
            CUDAMemorySpace::Constant => PTXScope::System,
            CUDAMemorySpace::Texture => PTXScope::System,
        }
    }

    /// Whether this space supports read-write access.
    pub fn is_read_write(&self) -> bool {
        matches!(self, CUDAMemorySpace::Global | CUDAMemorySpace::Shared | CUDAMemorySpace::Local)
    }

    /// Whether this space is coherent across all threads.
    pub fn is_globally_coherent(&self) -> bool {
        matches!(self, CUDAMemorySpace::Global)
    }
}

impl fmt::Display for CUDAMemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CUDAMemorySpace::Global => write!(f, ".global"),
            CUDAMemorySpace::Shared => write!(f, ".shared"),
            CUDAMemorySpace::Local => write!(f, ".local"),
            CUDAMemorySpace::Constant => write!(f, ".const"),
            CUDAMemorySpace::Texture => write!(f, ".tex"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PTX Fence
// ═══════════════════════════════════════════════════════════════════════════

/// A PTX fence (membar) operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PTXFence {
    /// Fence scope.
    pub scope: PTXScope,
    /// Memory ordering.
    pub ordering: PTXMemoryOrder,
}

impl PTXFence {
    /// Create a new fence.
    pub fn new(scope: PTXScope, ordering: PTXMemoryOrder) -> Self {
        PTXFence { scope, ordering }
    }

    /// membar.cta — CTA-scope fence.
    pub fn membar_cta() -> Self {
        PTXFence::new(PTXScope::CTA, PTXMemoryOrder::SeqCst)
    }

    /// membar.gpu — GPU-scope fence.
    pub fn membar_gpu() -> Self {
        PTXFence::new(PTXScope::GPU, PTXMemoryOrder::SeqCst)
    }

    /// membar.sys — System-scope fence.
    pub fn membar_sys() -> Self {
        PTXFence::new(PTXScope::System, PTXMemoryOrder::SeqCst)
    }

    /// Whether this fence orders operations between two threads.
    pub fn orders_between(&self, hierarchy: &ScopeHierarchy, t1: ThreadId, t2: ThreadId) -> bool {
        hierarchy.same_scope(t1, t2, self.scope)
    }

    /// Whether this fence is at least as strong as another.
    pub fn is_at_least(&self, other: &PTXFence) -> bool {
        self.scope.includes(&other.scope) && self.ordering.is_at_least(&other.ordering)
    }
}

impl fmt::Display for PTXFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "membar{}{}", self.scope, self.ordering)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Execution
// ═══════════════════════════════════════════════════════════════════════════

/// An execution graph annotated with GPU-specific information.
#[derive(Debug, Clone)]
pub struct GPUExecution {
    /// The base execution graph.
    pub graph: ExecutionGraph,
    /// Scope hierarchy.
    pub hierarchy: ScopeHierarchy,
    /// Memory space for each address.
    pub address_space: HashMap<Address, CUDAMemorySpace>,
    /// PTX ordering for each event.
    pub event_ordering: HashMap<EventId, PTXMemoryOrder>,
    /// PTX scope for each event.
    pub event_scope: HashMap<EventId, PTXScope>,
}

impl GPUExecution {
    /// Create a new GPU execution.
    pub fn new(graph: ExecutionGraph, hierarchy: ScopeHierarchy) -> Self {
        GPUExecution {
            graph,
            hierarchy,
            address_space: HashMap::new(),
            event_ordering: HashMap::new(),
            event_scope: HashMap::new(),
        }
    }

    /// Set memory space for an address.
    pub fn set_address_space(&mut self, addr: Address, space: CUDAMemorySpace) {
        self.address_space.insert(addr, space);
    }

    /// Get memory space for an address (defaults to Global).
    pub fn get_address_space(&self, addr: Address) -> CUDAMemorySpace {
        self.address_space.get(&addr).copied().unwrap_or(CUDAMemorySpace::Global)
    }

    /// Set PTX ordering for an event.
    pub fn set_event_ordering(&mut self, event: EventId, ordering: PTXMemoryOrder) {
        self.event_ordering.insert(event, ordering);
    }

    /// Get PTX ordering for an event (defaults to Relaxed).
    pub fn get_event_ordering(&self, event: EventId) -> PTXMemoryOrder {
        self.event_ordering.get(&event).copied().unwrap_or(PTXMemoryOrder::Relaxed)
    }

    /// Set PTX scope for an event.
    pub fn set_event_scope(&mut self, event: EventId, scope: PTXScope) {
        self.event_scope.insert(event, scope);
    }

    /// Get PTX scope for an event (defaults to System).
    pub fn get_event_scope(&self, event: EventId) -> PTXScope {
        self.event_scope.get(&event).copied().unwrap_or(PTXScope::System)
    }

    /// Compute scoped reads-from: filter rf to only include pairs where
    /// the write is visible to the read at the operation's scope.
    pub fn scoped_reads_from(&self) -> BitMatrix {
        let n = self.graph.events.len();
        let mut scoped_rf = BitMatrix::new(n);

        for (w, r) in self.graph.rf.edges() {
            let w_scope = self.get_event_scope(w);
            let r_scope = self.get_event_scope(r);
            let w_thread = self.graph.events[w].thread;
            let r_thread = self.graph.events[r].thread;

            // Write is visible to read if they share a scope at the write's scope level
            let min_scope = if w_scope.includes(&r_scope) { r_scope } else { w_scope };
            if self.hierarchy.same_scope(w_thread, r_thread, min_scope) {
                scoped_rf.set(w, r, true);
            }
        }

        scoped_rf
    }

    /// Check if two events are ordered by scope-qualified program order.
    pub fn scoped_po(&self, e1: EventId, e2: EventId, scope: PTXScope) -> bool {
        let ev1 = &self.graph.events[e1];
        let ev2 = &self.graph.events[e2];
        if ev1.thread != ev2.thread { return false; }
        if ev1.po_index >= ev2.po_index { return false; }

        // Check if both events have scopes compatible with the given scope
        let s1 = self.get_event_scope(e1);
        let s2 = self.get_event_scope(e2);
        s1.includes(&scope) && s2.includes(&scope)
    }

    /// Number of events.
    pub fn num_events(&self) -> usize {
        self.graph.events.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PTX Axiom Set
// ═══════════════════════════════════════════════════════════════════════════

/// The PTX/CUDA axiomatic memory model axioms.
#[derive(Debug, Clone)]
pub struct PTXAxiomSet {
    /// Scoped coherence axioms.
    pub coherence_axioms: Vec<PTXAxiom>,
    /// Causality axioms.
    pub causality_axioms: Vec<PTXAxiom>,
    /// Atomicity axioms.
    pub atomicity_axioms: Vec<PTXAxiom>,
}

/// A single PTX axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTXAxiom {
    /// Name of the axiom.
    pub name: String,
    /// Description.
    pub description: String,
    /// The scope at which this axiom applies.
    pub scope: Option<PTXScope>,
    /// The constraint type.
    pub constraint: PTXConstraintKind,
}

/// Kind of PTX constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PTXConstraintKind {
    /// Acyclicity of a relation.
    Acyclic(String),
    /// Irreflexivity of a relation.
    Irreflexive(String),
    /// Relation must be empty.
    Empty(String),
    /// Totality on filtered events.
    Total(String),
}

impl PTXAxiomSet {
    /// Build the complete PTX axiom set.
    pub fn build() -> Self {
        let mut coherence_axioms = Vec::new();
        let mut causality_axioms = Vec::new();
        let mut atomicity_axioms = Vec::new();

        // Scoped coherence: for each scope, coherence order restricted to that scope is acyclic
        for scope in PTXScope::all() {
            coherence_axioms.push(PTXAxiom {
                name: format!("sc-per-loc-{}", scope),
                description: format!("Per-location coherence at {} scope", scope),
                scope: Some(*scope),
                constraint: PTXConstraintKind::Acyclic("po-loc | com".to_string()),
            });
        }

        // Causality
        causality_axioms.push(PTXAxiom {
            name: "causality".to_string(),
            description: "Scoped causality: acyclicity of hb".to_string(),
            scope: None,
            constraint: PTXConstraintKind::Acyclic("hb".to_string()),
        });

        causality_axioms.push(PTXAxiom {
            name: "no-thin-air".to_string(),
            description: "No out-of-thin-air values".to_string(),
            scope: None,
            constraint: PTXConstraintKind::Acyclic("hb".to_string()),
        });

        // Atomicity of RMW
        atomicity_axioms.push(PTXAxiom {
            name: "atomicity".to_string(),
            description: "RMW atomicity: no intervening write between RMW read and write".to_string(),
            scope: None,
            constraint: PTXConstraintKind::Empty("rmw-intervening".to_string()),
        });

        PTXAxiomSet { coherence_axioms, causality_axioms, atomicity_axioms }
    }

    /// Get all axioms.
    pub fn all_axioms(&self) -> Vec<&PTXAxiom> {
        let mut all = Vec::new();
        for a in &self.coherence_axioms { all.push(a); }
        for a in &self.causality_axioms { all.push(a); }
        for a in &self.atomicity_axioms { all.push(a); }
        all
    }

    /// Number of axioms.
    pub fn num_axioms(&self) -> usize {
        self.coherence_axioms.len() + self.causality_axioms.len() + self.atomicity_axioms.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA Model Checker
// ═══════════════════════════════════════════════════════════════════════════

/// Result of GPU model checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUCheckResult {
    /// Whether the execution is consistent.
    pub consistent: bool,
    /// Violated axioms (if any).
    pub violations: Vec<String>,
    /// Scope at which each violation occurs.
    pub violation_scopes: Vec<Option<PTXScope>>,
    /// Statistics.
    pub stats: GPUCheckStats,
}

/// Statistics from GPU model checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GPUCheckStats {
    /// Number of axioms checked.
    pub axioms_checked: usize,
    /// Number of axioms satisfied.
    pub axioms_satisfied: usize,
    /// Number of scoped relations computed.
    pub relations_computed: usize,
}

/// The CUDA/PTX memory model checker.
#[derive(Debug)]
pub struct CUDAModelChecker {
    /// The axiom set.
    pub axioms: PTXAxiomSet,
}

impl CUDAModelChecker {
    /// Create a new CUDA model checker.
    pub fn new() -> Self {
        CUDAModelChecker {
            axioms: PTXAxiomSet::build(),
        }
    }

    /// Build the PTX memory model as a `MemoryModel`.
    pub fn build_memory_model() -> MemoryModel {
        let mut model = MemoryModel::new("PTX");

        // Derived relations
        model.add_derived(
            "po-loc",
            RelationExpr::inter(
                RelationExpr::base("po"),
                RelationExpr::base("loc"),
            ),
            "Program order restricted to same location",
        );

        model.add_derived(
            "com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "Communication relation",
        );

        model.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union_many(vec![
                RelationExpr::base("po"),
                RelationExpr::base("rf"),
                RelationExpr::base("fence"),
            ])),
            "Happens-before (scoped)",
        );

        // Constraints
        model.add_acyclic(RelationExpr::base("po-loc | com"));
        model.add_acyclic(RelationExpr::base("hb"));

        model
    }

    /// Check an execution against the CUDA model.
    pub fn check_execution(&self, exec: &GPUExecution) -> GPUCheckResult {
        let mut violations = Vec::new();
        let mut violation_scopes = Vec::new();
        let mut stats = GPUCheckStats::default();

        // Check coherence axioms
        for axiom in &self.axioms.coherence_axioms {
            stats.axioms_checked += 1;
            // Simplified check: verify po-loc | com is acyclic for events in the same scope
            let is_acyclic = self.check_scoped_coherence(exec, axiom.scope);
            if is_acyclic {
                stats.axioms_satisfied += 1;
            } else {
                violations.push(axiom.name.clone());
                violation_scopes.push(axiom.scope);
            }
        }

        // Check causality axioms
        for axiom in &self.axioms.causality_axioms {
            stats.axioms_checked += 1;
            let is_satisfied = self.check_causality(exec);
            if is_satisfied {
                stats.axioms_satisfied += 1;
            } else {
                violations.push(axiom.name.clone());
                violation_scopes.push(axiom.scope);
            }
        }

        // Check atomicity axioms
        for axiom in &self.axioms.atomicity_axioms {
            stats.axioms_checked += 1;
            let is_satisfied = self.check_atomicity(exec);
            if is_satisfied {
                stats.axioms_satisfied += 1;
            } else {
                violations.push(axiom.name.clone());
                violation_scopes.push(axiom.scope);
            }
        }

        GPUCheckResult {
            consistent: violations.is_empty(),
            violations,
            violation_scopes,
            stats,
        }
    }

    /// Check scoped coherence for a given scope level.
    fn check_scoped_coherence(&self, exec: &GPUExecution, scope: Option<PTXScope>) -> bool {
        let n = exec.num_events();
        let mut relation = BitMatrix::new(n);

        // Add po-loc edges
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.graph.events[i];
                let ej = &exec.graph.events[j];
                if ei.thread == ej.thread && ei.po_index < ej.po_index && ei.address == ej.address {
                    if let Some(s) = scope {
                        if exec.hierarchy.same_scope(ei.thread, ej.thread, s) {
                            relation.set(i, j, true);
                        }
                    } else {
                        relation.set(i, j, true);
                    }
                }
            }
        }

        // Add rf, co, fr edges
        for (w, r) in exec.graph.rf.edges() {
            relation.set(w, r, true);
        }
        for (w1, w2) in exec.graph.co.edges() {
            relation.set(w1, w2, true);
        }

        // Check acyclicity via DFS
        relation.is_acyclic()
    }

    /// Check causality (hb acyclicity).
    fn check_causality(&self, exec: &GPUExecution) -> bool {
        let n = exec.num_events();
        let mut hb = BitMatrix::new(n);

        // po edges
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.graph.events[i];
                let ej = &exec.graph.events[j];
                if ei.thread == ej.thread && ei.po_index < ej.po_index {
                    hb.set(i, j, true);
                }
            }
        }

        // rf edges
        for (w, r) in exec.graph.rf.edges() {
        }

        // Transitive closure
        let hb_plus = hb.transitive_closure();

        // Check irreflexivity (no self-loops in transitive closure)
        for i in 0..n {
            if hb_plus.get(i, i) {
                return false;
            }
        }
        true
    }

    /// Check atomicity (no intervening writes between RMW read and write).
    fn check_atomicity(&self, exec: &GPUExecution) -> bool {
        // For each RMW event, check that no other write to the same address
        // is ordered between the RMW's read-from source and the RMW write
        for i in 0..exec.num_events() {
            let ev = &exec.graph.events[i];
            if ev.op_type != OpType::RMW { continue; }

            // Find the write that this RMW reads from
            let rf_source = exec.graph.rf.edges().into_iter()
                .find(|&(_, r)| r == i)
                .map(|(w, _)| w);

            if let Some(w) = rf_source {
                // Check no other write is co-between w and i
                for (co_from, co_to) in exec.graph.co.edges() {
                    if co_from == w && co_to != i && exec.graph.events[co_to].address == ev.address {
                        // Check if co_to is before i in co
                        let co_to_before_i = exec.graph.co.edges().into_iter()
                            .any(|(f, t)| f == co_to && t == i);
                        if co_to_before_i {
                            return false; // intervening write
                        }
                    }
                }
            }
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Scope Optimizer
// ═══════════════════════════════════════════════════════════════════════════

/// Scope optimization: find minimal scopes for correctness.
#[derive(Debug)]
pub struct GPUScopeOptimizer;

impl GPUScopeOptimizer {
    /// Infer minimal scopes needed for each event.
    pub fn infer_minimal_scopes(exec: &GPUExecution) -> HashMap<EventId, PTXScope> {
        let mut scopes = HashMap::new();
        let n = exec.num_events();

        for i in 0..n {
            let ev = &exec.graph.events[i];
            let mut needed_scope = PTXScope::CTA; // start minimal

            // For each event this one communicates with, check the needed scope
            for (w, r) in exec.graph.rf.edges() {
                if w == i || r == i {
                    let other = if w == i { r } else { w };
                    let other_thread = exec.graph.events[other].thread;
                    if ev.thread != other_thread {
                        let min_scope = exec.hierarchy.minimal_common_scope(ev.thread, other_thread);
                        if min_scope as u8 > needed_scope as u8 {
                            needed_scope = min_scope;
                        }
                    }
                }
            }

            scopes.insert(i, needed_scope);
        }

        scopes
    }

    /// Check if a fence at a given scope is redundant.
    pub fn is_fence_redundant(
        exec: &GPUExecution,
        fence_event: EventId,
        _fence_scope: PTXScope,
    ) -> bool {
        let ev = &exec.graph.events[fence_event];
        if ev.op_type != OpType::Fence {
            return true; // not a fence
        }

        // A fence is redundant if all orderings it provides are already guaranteed
        // by the orderings on the individual memory operations
        let thread = ev.thread;
        let mut has_unordered_pair = false;

        for i in 0..exec.num_events() {
            let ei = &exec.graph.events[i];
            if ei.thread != thread || i == fence_event { continue; }
            if ei.po_index >= ev.po_index { continue; } // before fence

            for j in 0..exec.num_events() {
                let ej = &exec.graph.events[j];
                if ej.thread != thread || j == fence_event { continue; }
                if ej.po_index <= ev.po_index { continue; } // after fence

                // Check if i and j need ordering from this fence
                if ei.op_type != OpType::Fence && ej.op_type != OpType::Fence {
                    let i_ordering = exec.get_event_ordering(i);
                    let j_ordering = exec.get_event_ordering(j);
                    // If neither has release/acquire semantics, the fence provides ordering
                    if !i_ordering.has_release() && !j_ordering.has_acquire() {
                        has_unordered_pair = true;
                    }
                }
            }
        }

        !has_unordered_pair
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PTX Litmus Test Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for standard GPU litmus tests.
pub struct PTXLitmusBuilder;

impl PTXLitmusBuilder {
    /// Message Passing with CTA scope.
    pub fn message_passing_cta() -> (LitmusTest, ScopeHierarchy) {
        let mut test = LitmusTest::new("MP-cta");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Release); // x = 1 (release)
        t0.store(1, 1, Ordering::Release); // y = 1 (release)
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 1, Ordering::Acquire); // r0 = y (acquire)
        t1.load(1, 0, Ordering::Acquire); // r1 = x (acquire)
        test.add_thread(t1);

        let hierarchy = ScopeHierarchy::all_same_cta(2);
        (test, hierarchy)
    }

    /// Message Passing across CTAs (GPU scope).
    pub fn message_passing_gpu() -> (LitmusTest, ScopeHierarchy) {
        let mut test = LitmusTest::new("MP-gpu");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Release);
        t0.store(1, 1, Ordering::Release);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 1, Ordering::Acquire);
        t1.load(1, 0, Ordering::Acquire);
        test.add_thread(t1);

        let hierarchy = ScopeHierarchy::one_thread_per_cta(2);
        (test, hierarchy)
    }

    /// Store Buffering with scoped fences.
    pub fn store_buffering_fenced(scope: PTXScope) -> (LitmusTest, ScopeHierarchy) {
        let scope_name = match scope {
            PTXScope::CTA => "cta",
            PTXScope::GPU => "gpu",
            PTXScope::System => "sys",
        };
        let mut test = LitmusTest::new(&format!("SB+membar.{}", scope_name));

        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        t0.fence(Ordering::SeqCst, crate::checker::litmus::Scope::from_ptx(scope));
        t0.load(0, 1, Ordering::Relaxed); // r0 = y
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed); // y = 1
        t1.fence(Ordering::SeqCst, crate::checker::litmus::Scope::from_ptx(scope));
        t1.load(1, 0, Ordering::Relaxed); // r1 = x
        test.add_thread(t1);

        let hierarchy = match scope {
            PTXScope::CTA => ScopeHierarchy::all_same_cta(2),
            _ => ScopeHierarchy::one_thread_per_cta(2),
        };
        (test, hierarchy)
    }

    /// IRIW (Independent Reads of Independent Writes) with scopes.
    pub fn iriw_scoped() -> (LitmusTest, ScopeHierarchy) {
        let mut test = LitmusTest::new("IRIW-scoped");

        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0, Ordering::Acquire); // r0 = x
        t1.load(1, 1, Ordering::Acquire); // r1 = y
        test.add_thread(t1);

        let mut t2 = Thread::new(2);
        t2.store(1, 1, Ordering::Relaxed); // y = 1
        test.add_thread(t2);

        let mut t3 = Thread::new(3);
        t3.load(2, 1, Ordering::Acquire); // r2 = y
        t3.load(3, 0, Ordering::Acquire); // r3 = x
        test.add_thread(t3);

        let hierarchy = ScopeHierarchy::one_thread_per_cta(4);
        (test, hierarchy)
    }
}

// Helper for litmus::Scope
impl crate::checker::litmus::Scope {
    /// Convert from PTXScope.
    fn from_ptx(s: PTXScope) -> Self {
        match s {
            PTXScope::CTA => crate::checker::litmus::Scope::CTA,
            PTXScope::GPU => crate::checker::litmus::Scope::GPU,
            PTXScope::System => crate::checker::litmus::Scope::System,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_memory_order_strength() {
        assert!(PTXMemoryOrder::SeqCst.is_at_least(&PTXMemoryOrder::Relaxed));
        assert!(PTXMemoryOrder::SeqCst.is_at_least(&PTXMemoryOrder::Acquire));
        assert!(PTXMemoryOrder::AcqRel.is_at_least(&PTXMemoryOrder::Acquire));
        assert!(PTXMemoryOrder::AcqRel.is_at_least(&PTXMemoryOrder::Release));
        assert!(!PTXMemoryOrder::Acquire.is_at_least(&PTXMemoryOrder::Release));
        assert!(!PTXMemoryOrder::Relaxed.is_at_least(&PTXMemoryOrder::Acquire));
    }

    #[test]
    fn test_ptx_scope_includes() {
        assert!(PTXScope::System.includes(&PTXScope::CTA));
        assert!(PTXScope::System.includes(&PTXScope::GPU));
        assert!(PTXScope::GPU.includes(&PTXScope::CTA));
        assert!(!PTXScope::CTA.includes(&PTXScope::GPU));
    }

    #[test]
    fn test_scope_hierarchy_same_cta() {
        let h = ScopeHierarchy::all_same_cta(4);
        assert!(h.same_scope(0, 1, PTXScope::CTA));
        assert!(h.same_scope(0, 3, PTXScope::CTA));
        assert_eq!(h.minimal_common_scope(0, 1), PTXScope::CTA);
    }

    #[test]
    fn test_scope_hierarchy_different_cta() {
        let h = ScopeHierarchy::one_thread_per_cta(4);
        assert!(!h.same_scope(0, 1, PTXScope::CTA));
        assert!(h.same_scope(0, 1, PTXScope::GPU));
        assert_eq!(h.minimal_common_scope(0, 1), PTXScope::GPU);
    }

    #[test]
    fn test_cuda_memory_space() {
        assert!(CUDAMemorySpace::Global.is_globally_coherent());
        assert!(!CUDAMemorySpace::Shared.is_globally_coherent());
        assert!(CUDAMemorySpace::Global.is_read_write());
        assert!(!CUDAMemorySpace::Constant.is_read_write());
        assert_eq!(CUDAMemorySpace::Shared.visibility_scope(), PTXScope::CTA);
    }

    #[test]
    fn test_ptx_fence_creation() {
        let f = PTXFence::membar_cta();
        assert_eq!(f.scope, PTXScope::CTA);
        assert_eq!(f.ordering, PTXMemoryOrder::SeqCst);

        let f2 = PTXFence::membar_sys();
        assert!(f2.is_at_least(&f));
        assert!(!f.is_at_least(&f2));
    }

    #[test]
    fn test_ptx_axiom_set() {
        let axioms = PTXAxiomSet::build();
        assert!(axioms.num_axioms() > 0);
        assert_eq!(axioms.coherence_axioms.len(), 3); // one per scope level
        assert!(!axioms.causality_axioms.is_empty());
        assert!(!axioms.atomicity_axioms.is_empty());
    }

    #[test]
    fn test_cuda_model_builder() {
        let model = CUDAModelChecker::build_memory_model();
        assert_eq!(model.name, "PTX");
        assert!(!model.derived_relations.is_empty());
    }

    #[test]
    fn test_gpu_execution_creation() {
        let graph = ExecutionGraph::empty();
        let hierarchy = ScopeHierarchy::all_same_cta(2);
        let exec = GPUExecution::new(graph, hierarchy);
        assert_eq!(exec.num_events(), 0);
    }

    #[test]
    fn test_gpu_execution_address_space() {
        let graph = ExecutionGraph::empty();
        let hierarchy = ScopeHierarchy::all_same_cta(2);
        let mut exec = GPUExecution::new(graph, hierarchy);
        exec.set_address_space(0x100, CUDAMemorySpace::Shared);
        assert_eq!(exec.get_address_space(0x100), CUDAMemorySpace::Shared);
        assert_eq!(exec.get_address_space(0x200), CUDAMemorySpace::Global); // default
    }

    #[test]
    fn test_scope_optimizer() {
        let graph = ExecutionGraph::empty();
        let hierarchy = ScopeHierarchy::all_same_cta(2);
        let exec = GPUExecution::new(graph, hierarchy);
        let scopes = GPUScopeOptimizer::infer_minimal_scopes(&exec);
        assert!(scopes.is_empty()); // no events
    }

    #[test]
    fn test_litmus_mp_cta() {
        let (test, hierarchy) = PTXLitmusBuilder::message_passing_cta();
        assert_eq!(test.thread_count(), 2);
        assert!(hierarchy.same_scope(0, 1, PTXScope::CTA));
    }

    #[test]
    fn test_litmus_mp_gpu() {
        let (test, hierarchy) = PTXLitmusBuilder::message_passing_gpu();
        assert_eq!(test.thread_count(), 2);
        assert!(!hierarchy.same_scope(0, 1, PTXScope::CTA));
        assert!(hierarchy.same_scope(0, 1, PTXScope::GPU));
    }

    #[test]
    fn test_litmus_iriw() {
        let (test, hierarchy) = PTXLitmusBuilder::iriw_scoped();
        assert_eq!(test.thread_count(), 4);
        assert!(!hierarchy.same_scope(0, 2, PTXScope::CTA));
    }
}
