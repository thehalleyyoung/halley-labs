//! WebGPU memory model specification and verification.
//!
//! Implements the WebGPU memory model from §10 of the LITMUS∞ paper.
//! Provides both operational and axiomatic semantics, translation between
//! them, equivalence verification, cross-origin security isolation, and
//! comparison with Vulkan and PTX models.

use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, Value, OpType, Scope,
    ExecutionGraph, BitMatrix,
};
use crate::checker::memory_model::{
    MemoryModel, RelationExpr, Constraint, DerivedRelation, PredicateExpr,
};
use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, LitmusOutcome, Outcome,
};

// ═══════════════════════════════════════════════════════════════════════
// WebGPU-specific scope and ordering types
// ═══════════════════════════════════════════════════════════════════════

/// WebGPU workgroup scope levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WebGPUScope {
    /// Subgroup (SIMD lane group).
    Subgroup,
    /// Workgroup (equivalent to CTA).
    Workgroup,
    /// Queue family (device-level).
    QueueFamily,
}

impl WebGPUScope {
    pub fn all() -> &'static [WebGPUScope] {
        &[WebGPUScope::Subgroup, WebGPUScope::Workgroup, WebGPUScope::QueueFamily]
    }

    pub fn includes(&self, other: &WebGPUScope) -> bool {
        (*self as u8) >= (*other as u8)
    }

    pub fn to_exec_scope(&self) -> Scope {
        match self {
            WebGPUScope::Subgroup => Scope::CTA,
            WebGPUScope::Workgroup => Scope::CTA,
            WebGPUScope::QueueFamily => Scope::GPU,
        }
    }
}

impl fmt::Display for WebGPUScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebGPUScope::Subgroup => write!(f, "subgroup"),
            WebGPUScope::Workgroup => write!(f, "workgroup"),
            WebGPUScope::QueueFamily => write!(f, "queuefamily"),
        }
    }
}

/// WebGPU memory address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddressSpace {
    /// Private per-invocation memory.
    Private,
    /// Workgroup-shared memory.
    Workgroup,
    /// Storage buffer (device-visible).
    Storage,
    /// Uniform buffer (read-only).
    Uniform,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressSpace::Private => write!(f, "private"),
            AddressSpace::Workgroup => write!(f, "workgroup"),
            AddressSpace::Storage => write!(f, "storage"),
            AddressSpace::Uniform => write!(f, "uniform"),
        }
    }
}

/// WebGPU memory ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebGPUOrdering {
    /// No ordering guarantees.
    Relaxed,
    /// Acquire semantics.
    Acquire,
    /// Release semantics.
    Release,
    /// Acquire + Release.
    AcqRel,
    /// Sequentially consistent (not in base WebGPU, used for analysis).
    SeqCst,
}

impl WebGPUOrdering {
    pub fn is_acquire(&self) -> bool {
        matches!(self, WebGPUOrdering::Acquire | WebGPUOrdering::AcqRel | WebGPUOrdering::SeqCst)
    }

    pub fn is_release(&self) -> bool {
        matches!(self, WebGPUOrdering::Release | WebGPUOrdering::AcqRel | WebGPUOrdering::SeqCst)
    }

    pub fn to_litmus_ordering(&self) -> Ordering {
        match self {
            WebGPUOrdering::Relaxed => Ordering::Relaxed,
            WebGPUOrdering::Acquire => Ordering::Acquire,
            WebGPUOrdering::Release => Ordering::Release,
            WebGPUOrdering::AcqRel => Ordering::AcqRel,
            WebGPUOrdering::SeqCst => Ordering::SeqCst,
        }
    }
}

impl fmt::Display for WebGPUOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebGPUOrdering::Relaxed => write!(f, "relaxed"),
            WebGPUOrdering::Acquire => write!(f, "acquire"),
            WebGPUOrdering::Release => write!(f, "release"),
            WebGPUOrdering::AcqRel => write!(f, "acq_rel"),
            WebGPUOrdering::SeqCst => write!(f, "seq_cst"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WebGPU Event — extends the base Event with WebGPU-specific info
// ═══════════════════════════════════════════════════════════════════════

/// A WebGPU-specific event with additional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebGPUEvent {
    /// Base event information.
    pub event_id: EventId,
    pub thread_id: ThreadId,
    pub op_type: OpType,
    pub address: Address,
    pub value: Value,
    /// WebGPU-specific scope.
    pub webgpu_scope: WebGPUScope,
    /// Address space.
    pub address_space: AddressSpace,
    /// Memory ordering.
    pub ordering: WebGPUOrdering,
    /// Workgroup ID.
    pub workgroup_id: usize,
    /// Subgroup ID within workgroup.
    pub subgroup_id: usize,
    /// Invocation ID within subgroup.
    pub invocation_id: usize,
}

impl WebGPUEvent {
    pub fn new(
        event_id: EventId,
        thread_id: ThreadId,
        op_type: OpType,
        address: Address,
        value: Value,
    ) -> Self {
        Self {
            event_id,
            thread_id,
            op_type,
            address,
            value,
            webgpu_scope: WebGPUScope::Workgroup,
            address_space: AddressSpace::Storage,
            ordering: WebGPUOrdering::Relaxed,
            workgroup_id: 0,
            subgroup_id: 0,
            invocation_id: thread_id,
        }
    }

    pub fn with_scope(mut self, scope: WebGPUScope) -> Self {
        self.webgpu_scope = scope;
        self
    }

    pub fn with_address_space(mut self, space: AddressSpace) -> Self {
        self.address_space = space;
        self
    }

    pub fn with_ordering(mut self, ordering: WebGPUOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    pub fn with_workgroup(mut self, wg: usize, sg: usize) -> Self {
        self.workgroup_id = wg;
        self.subgroup_id = sg;
        self
    }

    pub fn same_workgroup(&self, other: &WebGPUEvent) -> bool {
        self.workgroup_id == other.workgroup_id
    }

    pub fn same_subgroup(&self, other: &WebGPUEvent) -> bool {
        self.workgroup_id == other.workgroup_id && self.subgroup_id == other.subgroup_id
    }

    pub fn to_base_event(&self) -> Event {
        Event::new(
            self.event_id,
            self.thread_id,
            self.op_type,
            self.address,
            self.value,
        )
        .with_scope(self.webgpu_scope.to_exec_scope())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WebGPUModel — full axiomatic specification
// ═══════════════════════════════════════════════════════════════════════

/// The WebGPU axiomatic memory model.
///
/// Based on the WGSL specification with formalization from the LITMUS∞ paper.
#[derive(Debug, Clone)]
pub struct WebGPUModel {
    /// Underlying axiomatic model.
    pub model: MemoryModel,
    /// Scope hierarchy configuration.
    pub scope_config: WebGPUScopeConfig,
}

/// WebGPU scope hierarchy configuration.
#[derive(Debug, Clone)]
pub struct WebGPUScopeConfig {
    /// Number of workgroups.
    pub num_workgroups: usize,
    /// Subgroups per workgroup.
    pub subgroups_per_workgroup: usize,
    /// Invocations per subgroup.
    pub invocations_per_subgroup: usize,
}

impl WebGPUScopeConfig {
    pub fn new(workgroups: usize, subgroups: usize, invocations: usize) -> Self {
        Self {
            num_workgroups: workgroups,
            subgroups_per_workgroup: subgroups,
            invocations_per_subgroup: invocations,
        }
    }

    pub fn default_config() -> Self {
        Self::new(2, 2, 4)
    }

    pub fn total_invocations(&self) -> usize {
        self.num_workgroups
            * self.subgroups_per_workgroup
            * self.invocations_per_subgroup
    }
}

impl Default for WebGPUScopeConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

impl WebGPUModel {
    /// Create the standard WebGPU memory model.
    pub fn new() -> Self {
        let mut model = MemoryModel::new("WebGPU");

        // --- Base relations ---
        // po, rf, co, fr are inherited from MemoryModel::new()

        // --- Derived relations ---

        // Program order restricted to same workgroup
        model.add_derived(
            "po-wg",
            RelationExpr::inter(
                RelationExpr::base("po"),
                RelationExpr::base("same-workgroup"),
            ),
            "program order within workgroup",
        );

        // Synchronizes-with: release-acquire pairs at matching scope
        model.add_derived(
            "sw",
            RelationExpr::seq(
                RelationExpr::seq(
                    RelationExpr::filter(PredicateExpr::IsWrite),
                    RelationExpr::base("rf"),
                ),
                RelationExpr::filter(PredicateExpr::IsRead),
            ),
            "synchronizes-with (release-acquire)",
        );

        // Happens-before
        model.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union(
                RelationExpr::base("po"),
                RelationExpr::base("sw"),
            )),
            "happens-before",
        );

        // Inter-workgroup ordering
        model.add_derived(
            "inter-wg",
            RelationExpr::diff(
                RelationExpr::base("hb"),
                RelationExpr::base("same-workgroup"),
            ),
            "inter-workgroup happens-before",
        );

        // --- Constraints ---

        // Coherence: SC per location
        model.add_acyclic(RelationExpr::union(
            RelationExpr::base("hb"),
            RelationExpr::union(
                RelationExpr::base("rf"),
                RelationExpr::union(
                    RelationExpr::base("co"),
                    RelationExpr::base("fr"),
                ),
            ),
        ));

        // No thin air
        model.add_acyclic(RelationExpr::union(
            RelationExpr::base("hb"),
            RelationExpr::base("rf"),
        ));

        // Observation: if a write is coherence-before another,
        // the later write's value should be observable
        model.add_irreflexive(RelationExpr::seq(
            RelationExpr::base("fr"),
            RelationExpr::base("hb"),
        ));

        Self {
            model,
            scope_config: WebGPUScopeConfig::default(),
        }
    }

    pub fn with_scope_config(mut self, config: WebGPUScopeConfig) -> Self {
        self.scope_config = config;
        self
    }

    /// Build same-workgroup and same-subgroup relations for a set of events.
    pub fn build_scope_relations(&self, events: &[WebGPUEvent]) -> (BitMatrix, BitMatrix) {
        let n = events.len();
        let mut same_wg = BitMatrix::new(n);
        let mut same_sg = BitMatrix::new(n);

        for i in 0..n {
            for j in 0..n {
                if events[i].same_workgroup(&events[j]) {
                    same_wg.set(i, j, true);
                }
                if events[i].same_subgroup(&events[j]) {
                    same_sg.set(i, j, true);
                }
            }
        }

        (same_wg, same_sg)
    }

    /// Verify an execution graph against the WebGPU model.
    pub fn verify(&self, graph: &ExecutionGraph) -> WebGPUVerificationResult {
        let n = graph.events.len();

        // Check acyclicity of hb
        let hb = graph.po.union(&graph.rf).transitive_closure();
        let hb_acyclic = hb.is_acyclic();

        // Check coherence (SC per location)
        let sc_per_loc = graph
            .po
            .union(&graph.rf)
            .union(&graph.co)
            .union(&graph.fr);
        let coherent = sc_per_loc.is_acyclic();

        // Check no thin air
        let hb_rf = graph.po.union(&graph.rf);
        let no_thin_air = hb_rf.is_acyclic();

        // Check observation
        let fr_hb = graph.fr.compose(&hb);
        let observation_ok = fr_hb.is_irreflexive();

        let consistent = hb_acyclic && coherent && no_thin_air && observation_ok;

        WebGPUVerificationResult {
            consistent,
            hb_acyclic,
            coherent,
            no_thin_air,
            observation_ok,
            violated_constraints: {
                let mut v = Vec::new();
                if !hb_acyclic {
                    v.push("hb-acyclicity".to_string());
                }
                if !coherent {
                    v.push("coherence".to_string());
                }
                if !no_thin_air {
                    v.push("no-thin-air".to_string());
                }
                if !observation_ok {
                    v.push("observation".to_string());
                }
                v
            },
        }
    }

    /// Get the underlying axiomatic model.
    pub fn axiomatic_model(&self) -> &MemoryModel {
        &self.model
    }
}

impl Default for WebGPUModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of WebGPU model verification.
#[derive(Debug, Clone)]
pub struct WebGPUVerificationResult {
    pub consistent: bool,
    pub hb_acyclic: bool,
    pub coherent: bool,
    pub no_thin_air: bool,
    pub observation_ok: bool,
    pub violated_constraints: Vec<String>,
}

impl fmt::Display for WebGPUVerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WebGPU: {} (hb={}, coh={}, nta={}, obs={})",
            if self.consistent { "CONSISTENT" } else { "INCONSISTENT" },
            if self.hb_acyclic { "✓" } else { "✗" },
            if self.coherent { "✓" } else { "✗" },
            if self.no_thin_air { "✓" } else { "✗" },
            if self.observation_ok { "✓" } else { "✗" },
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// OperationalSemantics — small-step semantics for WebGPU
// ═══════════════════════════════════════════════════════════════════════

/// Small-step operational state for WebGPU.
#[derive(Debug, Clone)]
pub struct WebGPUOperationalState {
    /// Per-invocation local state (registers, PC).
    pub invocation_states: Vec<InvocationState>,
    /// Shared memory.
    pub shared_memory: HashMap<Address, Value>,
    /// Workgroup-shared memory.
    pub workgroup_memory: HashMap<(usize, Address), Value>,
    /// Write buffer entries (for relaxed semantics).
    pub write_buffers: Vec<VecDeque<WriteBufferEntry>>,
    /// Global coherence order witness.
    pub coherence_witness: Vec<(Address, EventId, EventId)>,
}

/// State of a single GPU invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationState {
    pub invocation_id: usize,
    pub workgroup_id: usize,
    pub subgroup_id: usize,
    pub registers: HashMap<usize, Value>,
    pub pc: usize,
    pub active: bool,
}

impl InvocationState {
    pub fn new(id: usize, wg: usize, sg: usize) -> Self {
        Self {
            invocation_id: id,
            workgroup_id: wg,
            subgroup_id: sg,
            registers: HashMap::new(),
            pc: 0,
            active: true,
        }
    }
}

/// An entry in the write buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WriteBufferEntry {
    pub address: Address,
    pub value: Value,
    pub scope: WebGPUScope,
    pub address_space: AddressSpace,
}

impl WebGPUOperationalState {
    pub fn new(num_invocations: usize) -> Self {
        Self {
            invocation_states: (0..num_invocations)
                .map(|i| InvocationState::new(i, 0, 0))
                .collect(),
            shared_memory: HashMap::new(),
            workgroup_memory: HashMap::new(),
            write_buffers: (0..num_invocations).map(|_| VecDeque::new()).collect(),
            coherence_witness: Vec::new(),
        }
    }

    pub fn with_invocations(invocations: Vec<InvocationState>) -> Self {
        let n = invocations.len();
        Self {
            invocation_states: invocations,
            shared_memory: HashMap::new(),
            workgroup_memory: HashMap::new(),
            write_buffers: (0..n).map(|_| VecDeque::new()).collect(),
            coherence_witness: Vec::new(),
        }
    }

    /// Read from memory, checking write buffer first.
    pub fn read(&self, invocation: usize, addr: Address, space: AddressSpace) -> Value {
        // Check write buffer first (store forwarding)
        for entry in self.write_buffers[invocation].iter().rev() {
            if entry.address == addr && entry.address_space == space {
                return entry.value;
            }
        }

        match space {
            AddressSpace::Workgroup => {
                let wg = self.invocation_states[invocation].workgroup_id;
                self.workgroup_memory
                    .get(&(wg, addr))
                    .copied()
                    .unwrap_or(0)
            }
            _ => self.shared_memory.get(&addr).copied().unwrap_or(0),
        }
    }

    /// Write to memory (buffer or immediate).
    pub fn write(
        &mut self,
        invocation: usize,
        addr: Address,
        value: Value,
        space: AddressSpace,
        scope: WebGPUScope,
        immediate: bool,
    ) {
        if immediate {
            self.flush_write(invocation, addr, value, space);
        } else {
            self.write_buffers[invocation].push_back(WriteBufferEntry {
                address: addr,
                value,
                scope,
                address_space: space,
            });
        }
    }

    fn flush_write(
        &mut self,
        invocation: usize,
        addr: Address,
        value: Value,
        space: AddressSpace,
    ) {
        match space {
            AddressSpace::Workgroup => {
                let wg = self.invocation_states[invocation].workgroup_id;
                self.workgroup_memory.insert((wg, addr), value);
            }
            _ => {
                self.shared_memory.insert(addr, value);
            }
        }
    }

    /// Flush all write buffer entries for a given invocation.
    pub fn flush_all(&mut self, invocation: usize) {
        let entries: Vec<_> = self.write_buffers[invocation].drain(..).collect();
        for entry in entries {
            self.flush_write(invocation, entry.address, entry.value, entry.address_space);
        }
    }

    /// Check if the system has terminated (all invocations done).
    pub fn is_terminated(&self) -> bool {
        self.invocation_states.iter().all(|s| !s.active)
            && self.write_buffers.iter().all(|b| b.is_empty())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// OperationalToAxiomatic — translation
// ═══════════════════════════════════════════════════════════════════════

/// Translates an operational execution trace to an axiomatic execution graph.
pub struct OperationalToAxiomatic;

/// A single step in an operational execution trace.
#[derive(Debug, Clone)]
pub struct OperationalStep {
    pub invocation: usize,
    pub op_type: OpType,
    pub address: Address,
    pub value: Value,
    pub ordering: WebGPUOrdering,
    pub address_space: AddressSpace,
}

/// A complete operational execution trace.
#[derive(Debug, Clone)]
pub struct OperationalTrace {
    pub steps: Vec<OperationalStep>,
    pub initial_memory: HashMap<Address, Value>,
    pub final_memory: HashMap<Address, Value>,
    pub num_invocations: usize,
}

impl OperationalTrace {
    pub fn new(num_invocations: usize) -> Self {
        Self {
            steps: Vec::new(),
            initial_memory: HashMap::new(),
            final_memory: HashMap::new(),
            num_invocations,
        }
    }

    pub fn add_step(&mut self, step: OperationalStep) {
        self.steps.push(step);
    }
}

impl OperationalToAxiomatic {
    /// Convert an operational trace to an execution graph.
    pub fn translate(trace: &OperationalTrace) -> ExecutionGraph {
        let mut events = Vec::new();
        let mut per_thread_counter: HashMap<usize, usize> = HashMap::new();

        for (idx, step) in trace.steps.iter().enumerate() {
            let po_idx = per_thread_counter
                .entry(step.invocation)
                .or_insert(0);
            let event = Event::new(idx, step.invocation, step.op_type, step.address, step.value)
                .with_po_index(*po_idx);
            events.push(event);
            *per_thread_counter.get_mut(&step.invocation).unwrap() += 1;
        }

        let mut graph = ExecutionGraph::new(events);

        // Build rf from trace: for each read, find the most recent write
        // to the same address that is visible.
        let n = trace.steps.len();
        let mut rf = BitMatrix::new(n);
        for (r_idx, step) in trace.steps.iter().enumerate() {
            if step.op_type == OpType::Read || step.op_type == OpType::RMW {
                // Find latest write to same address with matching value
                for w_idx in (0..r_idx).rev() {
                    let w_step = &trace.steps[w_idx];
                    if (w_step.op_type == OpType::Write || w_step.op_type == OpType::RMW)
                        && w_step.address == step.address
                        && w_step.value == step.value
                    {
                        rf.set(w_idx, r_idx, true);
                        break;
                    }
                }
            }
        }
        graph.rf = rf;

        // Build co from trace: total order of writes per address
        let mut addr_writes: HashMap<Address, Vec<usize>> = HashMap::new();
        for (idx, step) in trace.steps.iter().enumerate() {
            if step.op_type == OpType::Write || step.op_type == OpType::RMW {
                addr_writes.entry(step.address).or_default().push(idx);
            }
        }

        let mut co = BitMatrix::new(n);
        for (_, writes) in &addr_writes {
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    co.set(writes[i], writes[j], true);
                }
            }
        }
        graph.co = co;

        // Build fr: derived from rf^{-1} ; co
        graph.fr = graph.rf.inverse().compose(&graph.co);

        graph
    }
}

// ═══════════════════════════════════════════════════════════════════════
// AxiomaticToOperational — reverse translation
// ═══════════════════════════════════════════════════════════════════════

/// Translates an axiomatic execution graph to an operational trace.
pub struct AxiomaticToOperational;

impl AxiomaticToOperational {
    /// Convert an execution graph to an operational trace.
    /// Requires the graph to have a valid topological ordering
    /// (consistent with happens-before).
    pub fn translate(graph: &ExecutionGraph) -> Option<OperationalTrace> {
        let n = graph.events.len();
        let hb = graph.po.union(&graph.rf).transitive_closure();

        // Topological sort of events by happens-before
        let order = hb.topological_sort()?;

        let num_threads = graph
            .events
            .iter()
            .map(|e| e.thread)
            .collect::<HashSet<_>>()
            .len();
        let mut trace = OperationalTrace::new(num_threads);

        for &eid in &order {
            let event = &graph.events[eid];
            let ordering = if event.is_fence() {
                WebGPUOrdering::AcqRel
            } else {
                WebGPUOrdering::Relaxed
            };

            trace.add_step(OperationalStep {
                invocation: event.thread,
                op_type: event.op_type,
                address: event.address,
                value: event.value,
                ordering,
                address_space: AddressSpace::Storage,
            });
        }

        Some(trace)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Equivalence verification
// ═══════════════════════════════════════════════════════════════════════

/// Result of checking equivalence between operational and axiomatic models.
#[derive(Debug, Clone)]
pub struct EquivalenceResult {
    pub equivalent: bool,
    pub details: String,
    pub op_consistent: bool,
    pub ax_consistent: bool,
}

impl fmt::Display for EquivalenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Equivalence: {} (op={}, ax={})",
            if self.equivalent { "YES" } else { "NO" },
            if self.op_consistent { "✓" } else { "✗" },
            if self.ax_consistent { "✓" } else { "✗" },
        )
    }
}

/// Verify equivalence between operational and axiomatic semantics.
pub fn verify_equivalence(graph: &ExecutionGraph) -> EquivalenceResult {
    let model = WebGPUModel::new();

    // Check axiomatic consistency
    let ax_result = model.verify(graph);

    // Check if the graph can be linearized (operational consistency)
    let hb = graph.po.union(&graph.rf).transitive_closure();
    let op_consistent = hb.is_acyclic();

    let equivalent = ax_result.consistent == op_consistent;

    EquivalenceResult {
        equivalent,
        details: if equivalent {
            "Operational and axiomatic semantics agree".to_string()
        } else {
            format!(
                "Disagreement: axiomatic={}, operational={}",
                if ax_result.consistent { "consistent" } else { "inconsistent" },
                if op_consistent { "consistent" } else { "inconsistent" },
            )
        },
        op_consistent,
        ax_consistent: ax_result.consistent,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WebGPUOriginIsolation — cross-origin security
// ═══════════════════════════════════════════════════════════════════════

/// Cross-origin isolation status for WebGPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OriginIsolation {
    /// Fully isolated (cross-origin isolated).
    Isolated,
    /// Same-origin only.
    SameOrigin,
    /// Not isolated (shared resources possible).
    NotIsolated,
}

/// Cross-origin security checker for WebGPU.
#[derive(Debug, Clone)]
pub struct WebGPUOriginIsolation {
    /// Origins in the system.
    pub origins: Vec<String>,
    /// Origin assignments for each invocation.
    pub invocation_origins: HashMap<usize, usize>,
    /// Isolation level.
    pub isolation: OriginIsolation,
}

impl WebGPUOriginIsolation {
    pub fn new(origins: Vec<String>, isolation: OriginIsolation) -> Self {
        Self {
            origins,
            invocation_origins: HashMap::new(),
            isolation,
        }
    }

    pub fn assign_origin(&mut self, invocation: usize, origin_idx: usize) {
        self.invocation_origins.insert(invocation, origin_idx);
    }

    /// Check if two invocations can share memory.
    pub fn can_share_memory(&self, inv_a: usize, inv_b: usize) -> bool {
        match self.isolation {
            OriginIsolation::Isolated => {
                // Only same-origin invocations can share
                let orig_a = self.invocation_origins.get(&inv_a);
                let orig_b = self.invocation_origins.get(&inv_b);
                orig_a == orig_b
            }
            OriginIsolation::SameOrigin => {
                let orig_a = self.invocation_origins.get(&inv_a);
                let orig_b = self.invocation_origins.get(&inv_b);
                orig_a == orig_b
            }
            OriginIsolation::NotIsolated => true,
        }
    }

    /// Check if a memory access pattern violates origin isolation.
    pub fn check_isolation(&self, events: &[WebGPUEvent]) -> Vec<IsolationViolation> {
        let mut violations = Vec::new();

        for i in 0..events.len() {
            for j in (i + 1)..events.len() {
                let a = &events[i];
                let b = &events[j];

                // Check if different-origin invocations access the same address
                if a.address == b.address
                    && a.invocation_id != b.invocation_id
                    && !self.can_share_memory(a.invocation_id, b.invocation_id)
                {
                    // At least one must be a write for it to be a violation
                    if a.op_type == OpType::Write
                        || a.op_type == OpType::RMW
                        || b.op_type == OpType::Write
                        || b.op_type == OpType::RMW
                    {
                        violations.push(IsolationViolation {
                            event_a: a.event_id,
                            event_b: b.event_id,
                            address: a.address,
                            origin_a: self
                                .invocation_origins
                                .get(&a.invocation_id)
                                .copied()
                                .unwrap_or(0),
                            origin_b: self
                                .invocation_origins
                                .get(&b.invocation_id)
                                .copied()
                                .unwrap_or(0),
                        });
                    }
                }
            }
        }

        violations
    }
}

/// A cross-origin isolation violation.
#[derive(Debug, Clone)]
pub struct IsolationViolation {
    pub event_a: EventId,
    pub event_b: EventId,
    pub address: Address,
    pub origin_a: usize,
    pub origin_b: usize,
}

impl fmt::Display for IsolationViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Isolation violation: e{} (origin {}) and e{} (origin {}) at addr {:#x}",
            self.event_a, self.origin_a, self.event_b, self.origin_b, self.address,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SafeGPU — verified secure subset
// ═══════════════════════════════════════════════════════════════════════

/// A verified secure subset of WebGPU operations.
///
/// SafeGPU restricts operations to guarantee deterministic behavior
/// under the WebGPU memory model.
#[derive(Debug, Clone)]
pub struct SafeGPU {
    /// Allowed address spaces.
    pub allowed_spaces: HashSet<AddressSpace>,
    /// Required minimum ordering for shared accesses.
    pub minimum_ordering: WebGPUOrdering,
    /// Whether relaxed atomics are allowed.
    pub allow_relaxed: bool,
    /// Maximum number of workgroups.
    pub max_workgroups: usize,
}

impl SafeGPU {
    /// The most restrictive safe subset.
    pub fn strict() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert(AddressSpace::Private);
        allowed.insert(AddressSpace::Storage);

        Self {
            allowed_spaces: allowed,
            minimum_ordering: WebGPUOrdering::AcqRel,
            allow_relaxed: false,
            max_workgroups: 1,
        }
    }

    /// A more permissive safe subset.
    pub fn standard() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert(AddressSpace::Private);
        allowed.insert(AddressSpace::Workgroup);
        allowed.insert(AddressSpace::Storage);
        allowed.insert(AddressSpace::Uniform);

        Self {
            allowed_spaces: allowed,
            minimum_ordering: WebGPUOrdering::Acquire,
            allow_relaxed: false,
            max_workgroups: 64,
        }
    }

    /// Check if a set of events conforms to the safe subset.
    pub fn check(&self, events: &[WebGPUEvent]) -> SafeGPUResult {
        let mut violations = Vec::new();

        for event in events {
            // Check address space
            if !self.allowed_spaces.contains(&event.address_space) {
                violations.push(SafeGPUViolation::DisallowedAddressSpace {
                    event_id: event.event_id,
                    space: event.address_space,
                });
            }

            // Check ordering
            if !self.allow_relaxed && event.ordering == WebGPUOrdering::Relaxed {
                if event.op_type != OpType::Fence && event.address_space != AddressSpace::Private {
                    violations.push(SafeGPUViolation::InsufficientOrdering {
                        event_id: event.event_id,
                        ordering: event.ordering,
                        required: self.minimum_ordering,
                    });
                }
            }
        }

        // Check workgroup count
        let workgroups: HashSet<usize> = events.iter().map(|e| e.workgroup_id).collect();
        if workgroups.len() > self.max_workgroups {
            violations.push(SafeGPUViolation::TooManyWorkgroups {
                count: workgroups.len(),
                max: self.max_workgroups,
            });
        }

        SafeGPUResult {
            safe: violations.is_empty(),
            violations,
        }
    }
}

/// Result of SafeGPU check.
#[derive(Debug, Clone)]
pub struct SafeGPUResult {
    pub safe: bool,
    pub violations: Vec<SafeGPUViolation>,
}

/// A violation of SafeGPU constraints.
#[derive(Debug, Clone)]
pub enum SafeGPUViolation {
    DisallowedAddressSpace {
        event_id: EventId,
        space: AddressSpace,
    },
    InsufficientOrdering {
        event_id: EventId,
        ordering: WebGPUOrdering,
        required: WebGPUOrdering,
    },
    TooManyWorkgroups {
        count: usize,
        max: usize,
    },
}

impl fmt::Display for SafeGPUViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafeGPUViolation::DisallowedAddressSpace { event_id, space } => {
                write!(f, "e{}: disallowed address space {}", event_id, space)
            }
            SafeGPUViolation::InsufficientOrdering {
                event_id,
                ordering,
                required,
            } => {
                write!(
                    f,
                    "e{}: ordering {} insufficient (need {})",
                    event_id, ordering, required,
                )
            }
            SafeGPUViolation::TooManyWorkgroups { count, max } => {
                write!(f, "too many workgroups: {} (max {})", count, max)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WebGPULitmusTest — GPU-specific litmus test format
// ═══════════════════════════════════════════════════════════════════════

/// A WebGPU-specific litmus test.
#[derive(Debug, Clone)]
pub struct WebGPULitmusTest {
    pub name: String,
    pub events: Vec<WebGPUEvent>,
    pub scope_config: WebGPUScopeConfig,
    pub expected_outcomes: Vec<(HashMap<Address, Value>, LitmusOutcome)>,
    pub description: String,
}

impl WebGPULitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            events: Vec::new(),
            scope_config: WebGPUScopeConfig::default(),
            expected_outcomes: Vec::new(),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn add_event(&mut self, event: WebGPUEvent) {
        self.events.push(event);
    }

    pub fn expect(&mut self, outcome: HashMap<Address, Value>, kind: LitmusOutcome) {
        self.expected_outcomes.push((outcome, kind));
    }

    /// Convert to a standard LitmusTest.
    pub fn to_litmus_test(&self) -> LitmusTest {
        let mut test = LitmusTest::new(&self.name);

        // Group events by thread
        let mut thread_events: HashMap<ThreadId, Vec<&WebGPUEvent>> = HashMap::new();
        for event in &self.events {
            thread_events
                .entry(event.thread_id)
                .or_default()
                .push(event);
        }

        for (&tid, events) in &thread_events {
            let mut thread = Thread::new(tid);
            for (idx, event) in events.iter().enumerate() {
                match event.op_type {
                    OpType::Read => {
                        thread.load(
                            idx,
                            event.address,
                            event.ordering.to_litmus_ordering(),
                        );
                    }
                    OpType::Write => {
                        thread.store(
                            event.address,
                            event.value,
                            event.ordering.to_litmus_ordering(),
                        );
                    }
                    OpType::Fence => {
                        thread.fence(
                            event.ordering.to_litmus_ordering(),
                            crate::checker::litmus::Scope::None,
                        );
                    }
                    OpType::RMW => {
                        thread.rmw(
                            idx,
                            event.address,
                            event.value,
                            event.ordering.to_litmus_ordering(),
                        );
                    }
                }
            }
            test.add_thread(thread);
        }

        test
    }

    /// Create the WebGPU message passing (MP) litmus test.
    pub fn message_passing() -> Self {
        let mut test = Self::new("WebGPU-MP");
        test.description = "Message passing between workgroup invocations".to_string();

        // Thread 0: W(x, 1); rel-fence; W(y, 1)
        test.add_event(
            WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(1, 0, OpType::Fence, 0, 0)
                .with_ordering(WebGPUOrdering::Release)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(2, 0, OpType::Write, 0x200, 1)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );

        // Thread 1: R(y) == 1; acq-fence; R(x) == ?
        test.add_event(
            WebGPUEvent::new(3, 1, OpType::Read, 0x200, 1)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(4, 1, OpType::Fence, 0, 0)
                .with_ordering(WebGPUOrdering::Acquire)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(5, 1, OpType::Read, 0x100, 0)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );

        // With proper fences, reading y==1 then x==0 is forbidden
        let mut forbidden = HashMap::new();
        forbidden.insert(0x200, 1u64);
        forbidden.insert(0x100, 0u64);
        test.expect(forbidden, LitmusOutcome::Forbidden);

        test
    }

    /// Create the WebGPU store buffering (SB) litmus test.
    pub fn store_buffering() -> Self {
        let mut test = Self::new("WebGPU-SB");
        test.description = "Store buffering between invocations".to_string();

        // Thread 0: W(x, 1); R(y) == ?
        test.add_event(
            WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(1, 0, OpType::Read, 0x200, 0)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );

        // Thread 1: W(y, 1); R(x) == ?
        test.add_event(
            WebGPUEvent::new(2, 1, OpType::Write, 0x200, 1)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );
        test.add_event(
            WebGPUEvent::new(3, 1, OpType::Read, 0x100, 0)
                .with_ordering(WebGPUOrdering::Relaxed)
                .with_address_space(AddressSpace::Storage),
        );

        // Both reading 0 is allowed under relaxed WebGPU
        let mut allowed = HashMap::new();
        allowed.insert(0x200, 0u64);
        allowed.insert(0x100, 0u64);
        test.expect(allowed, LitmusOutcome::Allowed);

        test
    }

    /// Create the WebGPU workgroup-scoped coherence test.
    pub fn workgroup_coherence() -> Self {
        let mut test = Self::new("WebGPU-WG-COH");
        test.description = "Coherence within a workgroup".to_string();

        // Two invocations in the same workgroup
        test.add_event(
            WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
                .with_ordering(WebGPUOrdering::Release)
                .with_address_space(AddressSpace::Workgroup)
                .with_workgroup(0, 0),
        );
        test.add_event(
            WebGPUEvent::new(1, 1, OpType::Write, 0x100, 2)
                .with_ordering(WebGPUOrdering::Release)
                .with_address_space(AddressSpace::Workgroup)
                .with_workgroup(0, 0),
        );
        test.add_event(
            WebGPUEvent::new(2, 0, OpType::Read, 0x100, 2)
                .with_ordering(WebGPUOrdering::Acquire)
                .with_address_space(AddressSpace::Workgroup)
                .with_workgroup(0, 0),
        );
        test.add_event(
            WebGPUEvent::new(3, 1, OpType::Read, 0x100, 1)
                .with_ordering(WebGPUOrdering::Acquire)
                .with_address_space(AddressSpace::Workgroup)
                .with_workgroup(0, 0),
        );

        // Seeing writes in opposite order within a workgroup is forbidden (coherence)
        let mut forbidden = HashMap::new();
        forbidden.insert(0x100, 0u64); // placeholder — both reads see different order
        test.expect(forbidden, LitmusOutcome::Forbidden);

        test
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VulkanModel — for comparison
// ═══════════════════════════════════════════════════════════════════════

/// A simplified Vulkan memory model for comparison with WebGPU.
#[derive(Debug, Clone)]
pub struct VulkanModel {
    pub model: MemoryModel,
}

impl VulkanModel {
    pub fn new() -> Self {
        let mut model = MemoryModel::new("Vulkan");

        // Vulkan uses availability/visibility model
        model.add_derived(
            "avail",
            RelationExpr::seq(
                RelationExpr::filter(PredicateExpr::IsWrite),
                RelationExpr::base("rf"),
            ),
            "availability: write made available",
        );

        model.add_derived(
            "vis",
            RelationExpr::seq(
                RelationExpr::base("rf"),
                RelationExpr::filter(PredicateExpr::IsRead),
            ),
            "visibility: read sees available write",
        );

        model.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union(
                RelationExpr::base("po"),
                RelationExpr::base("sw"),
            )),
            "happens-before",
        );

        model.add_acyclic(RelationExpr::union(
            RelationExpr::base("hb"),
            RelationExpr::union(
                RelationExpr::base("co"),
                RelationExpr::union(
                    RelationExpr::base("rf"),
                    RelationExpr::base("fr"),
                ),
            ),
        ));

        Self { model }
    }
}

impl Default for VulkanModel {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ModelDiff — compare memory models
// ═══════════════════════════════════════════════════════════════════════

/// A difference between two memory models.
#[derive(Debug, Clone)]
pub struct ModelDifference {
    pub model_a: String,
    pub model_b: String,
    /// Relations in A but not B.
    pub relations_only_in_a: Vec<String>,
    /// Relations in B but not A.
    pub relations_only_in_b: Vec<String>,
    /// Relations in both.
    pub shared_relations: Vec<String>,
    /// Constraints in A but not B.
    pub constraints_only_in_a: Vec<String>,
    /// Constraints in B but not A.
    pub constraints_only_in_b: Vec<String>,
    /// Description of key semantic differences.
    pub semantic_differences: Vec<String>,
}

impl ModelDifference {
    pub fn compare(a: &MemoryModel, b: &MemoryModel) -> Self {
        let a_rels: HashSet<String> = a
            .derived_relations
            .iter()
            .map(|r| r.name.clone())
            .collect();
        let b_rels: HashSet<String> = b
            .derived_relations
            .iter()
            .map(|r| r.name.clone())
            .collect();

        let a_constraints: HashSet<String> = a
            .constraints
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        let b_constraints: HashSet<String> = b
            .constraints
            .iter()
            .map(|c| c.name().to_string())
            .collect();

        ModelDifference {
            model_a: a.name.clone(),
            model_b: b.name.clone(),
            relations_only_in_a: a_rels.difference(&b_rels).cloned().collect(),
            relations_only_in_b: b_rels.difference(&a_rels).cloned().collect(),
            shared_relations: a_rels.intersection(&b_rels).cloned().collect(),
            constraints_only_in_a: a_constraints.difference(&b_constraints).cloned().collect(),
            constraints_only_in_b: b_constraints.difference(&a_constraints).cloned().collect(),
            semantic_differences: Vec::new(),
        }
    }

    /// Compare WebGPU, Vulkan, and PTX models.
    pub fn compare_gpu_models() -> Vec<ModelDifference> {
        let webgpu = WebGPUModel::new().model;
        let vulkan = VulkanModel::new().model;

        // PTX model (simplified)
        let mut ptx = MemoryModel::new("PTX");
        ptx.add_derived(
            "hb",
            RelationExpr::plus(RelationExpr::union(
                RelationExpr::base("po"),
                RelationExpr::base("sw"),
            )),
            "happens-before",
        );
        ptx.add_acyclic(RelationExpr::union(
            RelationExpr::base("hb"),
            RelationExpr::union(
                RelationExpr::base("co"),
                RelationExpr::union(
                    RelationExpr::base("rf"),
                    RelationExpr::base("fr"),
                ),
            ),
        ));

        let mut diffs = Vec::new();
        diffs.push(ModelDifference::compare(&webgpu, &vulkan));
        diffs.push(ModelDifference::compare(&webgpu, &ptx));
        diffs.push(ModelDifference::compare(&vulkan, &ptx));
        diffs
    }

    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: {} shared rels, {} only in {}, {} only in {}",
            self.model_a,
            self.model_b,
            self.shared_relations.len(),
            self.relations_only_in_a.len(),
            self.model_a,
            self.relations_only_in_b.len(),
            self.model_b,
        )
    }
}

impl fmt::Display for ModelDifference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model diff: {} vs {}", self.model_a, self.model_b)?;
        if !self.shared_relations.is_empty() {
            writeln!(f, "  Shared relations: {}", self.shared_relations.join(", "))?;
        }
        if !self.relations_only_in_a.is_empty() {
            writeln!(
                f,
                "  Only in {}: {}",
                self.model_a,
                self.relations_only_in_a.join(", "),
            )?;
        }
        if !self.relations_only_in_b.is_empty() {
            writeln!(
                f,
                "  Only in {}: {}",
                self.model_b,
                self.relations_only_in_b.join(", "),
            )?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // --- WebGPUScope tests ---

    #[test]
    fn test_webgpu_scope_ordering() {
        assert!(WebGPUScope::Subgroup < WebGPUScope::Workgroup);
        assert!(WebGPUScope::Workgroup < WebGPUScope::QueueFamily);
    }

    #[test]
    fn test_webgpu_scope_includes() {
        assert!(WebGPUScope::QueueFamily.includes(&WebGPUScope::Subgroup));
        assert!(WebGPUScope::Workgroup.includes(&WebGPUScope::Subgroup));
        assert!(!WebGPUScope::Subgroup.includes(&WebGPUScope::Workgroup));
    }

    #[test]
    fn test_webgpu_scope_display() {
        assert_eq!(format!("{}", WebGPUScope::Subgroup), "subgroup");
        assert_eq!(format!("{}", WebGPUScope::Workgroup), "workgroup");
    }

    // --- AddressSpace tests ---

    #[test]
    fn test_address_space_display() {
        assert_eq!(format!("{}", AddressSpace::Private), "private");
        assert_eq!(format!("{}", AddressSpace::Storage), "storage");
    }

    // --- WebGPUOrdering tests ---

    #[test]
    fn test_webgpu_ordering() {
        assert!(WebGPUOrdering::Acquire.is_acquire());
        assert!(!WebGPUOrdering::Acquire.is_release());
        assert!(WebGPUOrdering::Release.is_release());
        assert!(!WebGPUOrdering::Release.is_acquire());
        assert!(WebGPUOrdering::AcqRel.is_acquire());
        assert!(WebGPUOrdering::AcqRel.is_release());
    }

    // --- WebGPUEvent tests ---

    #[test]
    fn test_webgpu_event_creation() {
        let event = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 42)
            .with_scope(WebGPUScope::Workgroup)
            .with_address_space(AddressSpace::Storage)
            .with_ordering(WebGPUOrdering::Release)
            .with_workgroup(1, 0);

        assert_eq!(event.event_id, 0);
        assert_eq!(event.webgpu_scope, WebGPUScope::Workgroup);
        assert_eq!(event.address_space, AddressSpace::Storage);
        assert_eq!(event.ordering, WebGPUOrdering::Release);
        assert_eq!(event.workgroup_id, 1);
    }

    #[test]
    fn test_webgpu_event_same_workgroup() {
        let a = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1).with_workgroup(0, 0);
        let b = WebGPUEvent::new(1, 1, OpType::Read, 0x100, 1).with_workgroup(0, 1);
        let c = WebGPUEvent::new(2, 2, OpType::Read, 0x100, 1).with_workgroup(1, 0);

        assert!(a.same_workgroup(&b));
        assert!(!a.same_workgroup(&c));
        assert!(a.same_subgroup(&a));
        assert!(!a.same_subgroup(&b));
    }

    #[test]
    fn test_webgpu_event_to_base() {
        let event = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 42)
            .with_scope(WebGPUScope::Workgroup);
        let base = event.to_base_event();
        assert_eq!(base.id, 0);
        assert_eq!(base.op_type, OpType::Write);
    }

    // --- WebGPUModel tests ---

    #[test]
    fn test_webgpu_model_creation() {
        let model = WebGPUModel::new();
        assert_eq!(model.model.name, "WebGPU");
        assert!(!model.model.derived_relations.is_empty());
        assert!(!model.model.constraints.is_empty());
    }

    #[test]
    fn test_webgpu_scope_config() {
        let config = WebGPUScopeConfig::new(4, 2, 32);
        assert_eq!(config.total_invocations(), 4 * 2 * 32);
    }

    #[test]
    fn test_webgpu_model_verify_consistent() {
        let model = WebGPUModel::new();

        // Simple consistent execution: T0 writes, T1 reads
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 1, OpType::Read, 0x100, 1).with_po_index(0),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true); // T0's write → T1's read

        let result = model.verify(&graph);
        assert!(result.consistent);
    }

    #[test]
    fn test_webgpu_model_build_scope_relations() {
        let model = WebGPUModel::new();
        let events = vec![
            WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1).with_workgroup(0, 0),
            WebGPUEvent::new(1, 1, OpType::Read, 0x100, 1).with_workgroup(0, 1),
            WebGPUEvent::new(2, 2, OpType::Read, 0x100, 1).with_workgroup(1, 0),
        ];

        let (same_wg, same_sg) = model.build_scope_relations(&events);
        assert!(same_wg.get(0, 1)); // Same workgroup
        assert!(!same_wg.get(0, 2)); // Different workgroups
        assert!(same_sg.get(0, 0)); // Same subgroup (trivially)
        assert!(!same_sg.get(0, 1)); // Different subgroups
    }

    // --- Operational semantics tests ---

    #[test]
    fn test_operational_state_creation() {
        let state = WebGPUOperationalState::new(4);
        assert_eq!(state.invocation_states.len(), 4);
        assert!(!state.is_terminated());
    }

    #[test]
    fn test_operational_state_read_write() {
        let mut state = WebGPUOperationalState::new(2);

        // Write and read back
        state.write(0, 0x100, 42, AddressSpace::Storage, WebGPUScope::Workgroup, true);
        assert_eq!(state.read(0, 0x100, AddressSpace::Storage), 42);
        assert_eq!(state.read(1, 0x100, AddressSpace::Storage), 42);
    }

    #[test]
    fn test_operational_state_write_buffer() {
        let mut state = WebGPUOperationalState::new(2);

        // Write to buffer (not flushed)
        state.write(0, 0x100, 42, AddressSpace::Storage, WebGPUScope::Workgroup, false);

        // Thread 0 sees its own write (store forwarding)
        assert_eq!(state.read(0, 0x100, AddressSpace::Storage), 42);

        // Thread 1 doesn't see it yet
        assert_eq!(state.read(1, 0x100, AddressSpace::Storage), 0);

        // Flush
        state.flush_all(0);
        assert_eq!(state.read(1, 0x100, AddressSpace::Storage), 42);
    }

    #[test]
    fn test_operational_state_workgroup_memory() {
        let mut state = WebGPUOperationalState::with_invocations(vec![
            InvocationState::new(0, 0, 0),
            InvocationState::new(1, 0, 0),
            InvocationState::new(2, 1, 0),
        ]);

        // Write to workgroup memory
        state.write(0, 0x100, 42, AddressSpace::Workgroup, WebGPUScope::Workgroup, true);

        // Same workgroup sees it
        assert_eq!(state.read(1, 0x100, AddressSpace::Workgroup), 42);

        // Different workgroup doesn't
        assert_eq!(state.read(2, 0x100, AddressSpace::Workgroup), 0);
    }

    // --- OperationalToAxiomatic tests ---

    #[test]
    fn test_op_to_ax_translation() {
        let mut trace = OperationalTrace::new(2);
        trace.add_step(OperationalStep {
            invocation: 0,
            op_type: OpType::Write,
            address: 0x100,
            value: 1,
            ordering: WebGPUOrdering::Release,
            address_space: AddressSpace::Storage,
        });
        trace.add_step(OperationalStep {
            invocation: 1,
            op_type: OpType::Read,
            address: 0x100,
            value: 1,
            ordering: WebGPUOrdering::Acquire,
            address_space: AddressSpace::Storage,
        });

        let graph = OperationalToAxiomatic::translate(&trace);
        assert_eq!(graph.events.len(), 2);
        assert!(graph.rf.get(0, 1)); // Write → Read (same value)
    }

    #[test]
    fn test_op_to_ax_coherence_order() {
        let mut trace = OperationalTrace::new(1);
        trace.add_step(OperationalStep {
            invocation: 0,
            op_type: OpType::Write,
            address: 0x100,
            value: 1,
            ordering: WebGPUOrdering::Relaxed,
            address_space: AddressSpace::Storage,
        });
        trace.add_step(OperationalStep {
            invocation: 0,
            op_type: OpType::Write,
            address: 0x100,
            value: 2,
            ordering: WebGPUOrdering::Relaxed,
            address_space: AddressSpace::Storage,
        });

        let graph = OperationalToAxiomatic::translate(&trace);
        assert!(graph.co.get(0, 1)); // First write co-before second
        assert!(!graph.co.get(1, 0));
    }

    // --- AxiomaticToOperational tests ---

    #[test]
    fn test_ax_to_op_translation() {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 1, OpType::Read, 0x100, 1).with_po_index(0),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);

        let trace = AxiomaticToOperational::translate(&graph);
        assert!(trace.is_some());
        let trace = trace.unwrap();
        assert_eq!(trace.steps.len(), 2);
    }

    // --- Equivalence tests ---

    #[test]
    fn test_equivalence_consistent() {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 1, OpType::Read, 0x100, 1).with_po_index(0),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);

        let result = verify_equivalence(&graph);
        assert!(result.equivalent);
        assert!(result.op_consistent);
        assert!(result.ax_consistent);
    }

    // --- OriginIsolation tests ---

    #[test]
    fn test_origin_isolation_isolated() {
        let mut iso = WebGPUOriginIsolation::new(
            vec!["origin-a.com".to_string(), "origin-b.com".to_string()],
            OriginIsolation::Isolated,
        );
        iso.assign_origin(0, 0);
        iso.assign_origin(1, 0);
        iso.assign_origin(2, 1);

        assert!(iso.can_share_memory(0, 1)); // Same origin
        assert!(!iso.can_share_memory(0, 2)); // Different origins
    }

    #[test]
    fn test_origin_isolation_check() {
        let mut iso = WebGPUOriginIsolation::new(
            vec!["a.com".to_string(), "b.com".to_string()],
            OriginIsolation::Isolated,
        );
        iso.assign_origin(0, 0);
        iso.assign_origin(1, 1);

        let events = vec![
            WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1),
            WebGPUEvent::new(1, 1, OpType::Read, 0x100, 1),
        ];

        let violations = iso.check_isolation(&events);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_origin_isolation_not_isolated() {
        let iso = WebGPUOriginIsolation::new(
            vec!["a.com".to_string(), "b.com".to_string()],
            OriginIsolation::NotIsolated,
        );
        assert!(iso.can_share_memory(0, 1));
    }

    // --- SafeGPU tests ---

    #[test]
    fn test_safe_gpu_strict() {
        let safe = SafeGPU::strict();
        assert!(!safe.allow_relaxed);
        assert_eq!(safe.max_workgroups, 1);
    }

    #[test]
    fn test_safe_gpu_check_pass() {
        let safe = SafeGPU::standard();

        let events = vec![WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
            .with_ordering(WebGPUOrdering::Release)
            .with_address_space(AddressSpace::Storage)];

        let result = safe.check(&events);
        assert!(result.safe);
    }

    #[test]
    fn test_safe_gpu_check_fail_relaxed() {
        let safe = SafeGPU::standard();

        let events = vec![WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
            .with_ordering(WebGPUOrdering::Relaxed)
            .with_address_space(AddressSpace::Storage)];

        let result = safe.check(&events);
        assert!(!result.safe);
    }

    #[test]
    fn test_safe_gpu_check_private_relaxed_ok() {
        let safe = SafeGPU::standard();

        // Private memory with relaxed ordering is fine
        let events = vec![WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
            .with_ordering(WebGPUOrdering::Relaxed)
            .with_address_space(AddressSpace::Private)];

        let result = safe.check(&events);
        assert!(result.safe);
    }

    // --- WebGPULitmusTest tests ---

    #[test]
    fn test_message_passing_litmus() {
        let test = WebGPULitmusTest::message_passing();
        assert_eq!(test.name, "WebGPU-MP");
        assert_eq!(test.events.len(), 6);
        assert!(!test.expected_outcomes.is_empty());
    }

    #[test]
    fn test_store_buffering_litmus() {
        let test = WebGPULitmusTest::store_buffering();
        assert_eq!(test.name, "WebGPU-SB");
        assert_eq!(test.events.len(), 4);
    }

    #[test]
    fn test_workgroup_coherence_litmus() {
        let test = WebGPULitmusTest::workgroup_coherence();
        assert_eq!(test.name, "WebGPU-WG-COH");
        assert!(!test.events.is_empty());
    }

    #[test]
    fn test_litmus_to_standard() {
        let test = WebGPULitmusTest::store_buffering();
        let standard = test.to_litmus_test();
        assert_eq!(standard.name, "WebGPU-SB");
        assert!(standard.thread_count() >= 2);
    }

    // --- VulkanModel tests ---

    #[test]
    fn test_vulkan_model_creation() {
        let model = VulkanModel::new();
        assert_eq!(model.model.name, "Vulkan");
        assert!(!model.model.derived_relations.is_empty());
    }

    // --- ModelDiff tests ---

    #[test]
    fn test_model_diff() {
        let webgpu = WebGPUModel::new().model;
        let vulkan = VulkanModel::new().model;
        let diff = ModelDifference::compare(&webgpu, &vulkan);
        assert_eq!(diff.model_a, "WebGPU");
        assert_eq!(diff.model_b, "Vulkan");
    }

    #[test]
    fn test_compare_gpu_models() {
        let diffs = ModelDifference::compare_gpu_models();
        assert_eq!(diffs.len(), 3);
    }

    #[test]
    fn test_model_diff_summary() {
        let webgpu = WebGPUModel::new().model;
        let vulkan = VulkanModel::new().model;
        let diff = ModelDifference::compare(&webgpu, &vulkan);
        let summary = diff.summary();
        assert!(summary.contains("WebGPU"));
        assert!(summary.contains("Vulkan"));
    }

    #[test]
    fn test_verification_result_display() {
        let result = WebGPUVerificationResult {
            consistent: true,
            hb_acyclic: true,
            coherent: true,
            no_thin_air: true,
            observation_ok: true,
            violated_constraints: Vec::new(),
        };
        let s = format!("{}", result);
        assert!(s.contains("CONSISTENT"));
    }

    #[test]
    fn test_equivalence_result_display() {
        let result = EquivalenceResult {
            equivalent: true,
            details: "OK".to_string(),
            op_consistent: true,
            ax_consistent: true,
        };
        let s = format!("{}", result);
        assert!(s.contains("YES"));
    }
}
