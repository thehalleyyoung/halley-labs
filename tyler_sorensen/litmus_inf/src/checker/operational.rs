//! Operational semantics for memory models.
//!
//! Implements small-step operational semantics from §10 of the LITMUS∞ paper.
//! Provides `MachineState` with per-thread states, shared memory, and buffers.
//! Supports TSO (store buffers), ARM/RISC-V (write buffers), and GPU
//! (scope hierarchy) models. Includes operational ↔ axiomatic equivalence checking.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, Value, OpType, Scope,
    ExecutionGraph, BitMatrix,
};

// ═══════════════════════════════════════════════════════════════════════
// MemoryModelKind — which operational model to use
// ═══════════════════════════════════════════════════════════════════════

/// The kind of operational memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryModelKind {
    /// Sequential Consistency — no reordering.
    SC,
    /// Total Store Order (x86-TSO).
    TSO,
    /// Partial Store Order.
    PSO,
    /// ARM-style relaxed model with write buffers.
    ARM,
    /// RISC-V RVWMO.
    RISCV,
    /// GPU scoped model.
    GPU,
}

impl fmt::Display for MemoryModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryModelKind::SC => write!(f, "SC"),
            MemoryModelKind::TSO => write!(f, "TSO"),
            MemoryModelKind::PSO => write!(f, "PSO"),
            MemoryModelKind::ARM => write!(f, "ARM"),
            MemoryModelKind::RISCV => write!(f, "RISC-V"),
            MemoryModelKind::GPU => write!(f, "GPU"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ThreadState — per-thread state
// ═══════════════════════════════════════════════════════════════════════

/// State of a single thread in the operational model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadState {
    pub thread_id: ThreadId,
    /// Register file.
    pub registers: HashMap<usize, Value>,
    /// Program counter (index into instruction stream).
    pub pc: usize,
    /// Whether the thread has terminated.
    pub terminated: bool,
    /// Scope assignment for GPU models.
    pub scope: Scope,
}

impl ThreadState {
    pub fn new(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            registers: HashMap::new(),
            pc: 0,
            terminated: false,
            scope: Scope::None,
        }
    }

    pub fn with_scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    pub fn read_register(&self, reg: usize) -> Value {
        self.registers.get(&reg).copied().unwrap_or(0)
    }

    pub fn write_register(&mut self, reg: usize, value: Value) {
        self.registers.insert(reg, value);
    }

    pub fn advance_pc(&mut self) {
        self.pc += 1;
    }

    pub fn set_pc(&mut self, pc: usize) {
        self.pc = pc;
    }

    pub fn terminate(&mut self) {
        self.terminated = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// StoreBuffer — TSO store buffer
// ═══════════════════════════════════════════════════════════════════════

/// A FIFO store buffer entry (used by TSO and PSO models).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreBufferEntry {
    pub address: Address,
    pub value: Value,
    pub timestamp: u64,
}

/// A per-thread store buffer (FIFO) for TSO/PSO models.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreBuffer {
    pub thread_id: ThreadId,
    pub entries: VecDeque<StoreBufferEntry>,
    pub max_size: usize,
    next_timestamp: u64,
}

impl StoreBuffer {
    pub fn new(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            entries: VecDeque::new(),
            max_size: 64,
            next_timestamp: 0,
        }
    }

    pub fn with_max_size(mut self, max: usize) -> Self {
        self.max_size = max;
        self
    }

    /// Add a store to the buffer.
    pub fn push(&mut self, address: Address, value: Value) {
        self.entries.push_back(StoreBufferEntry {
            address,
            value,
            timestamp: self.next_timestamp,
        });
        self.next_timestamp += 1;
    }

    /// Drain the oldest entry to memory.
    pub fn drain_oldest(&mut self) -> Option<StoreBufferEntry> {
        self.entries.pop_front()
    }

    /// Look up the most recent buffered store to a given address.
    pub fn lookup(&self, address: Address) -> Option<Value> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.address == address)
            .map(|e| e.value)
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_size
    }

    /// Number of buffered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Flush all entries for a given address.
    pub fn flush_address(&mut self, address: Address) -> Vec<StoreBufferEntry> {
        let mut flushed = Vec::new();
        let mut remaining = VecDeque::new();
        for entry in self.entries.drain(..) {
            if entry.address == address {
                flushed.push(entry);
            } else {
                remaining.push_back(entry);
            }
        }
        self.entries = remaining;
        flushed
    }

    /// Flush all entries.
    pub fn flush_all(&mut self) -> Vec<StoreBufferEntry> {
        self.entries.drain(..).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WriteBuffer — ARM/RISC-V write buffer
// ═══════════════════════════════════════════════════════════════════════

/// A write buffer entry with scope information (for ARM/RISC-V/GPU).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WriteBufferEntry {
    pub address: Address,
    pub value: Value,
    pub scope: Scope,
    pub timestamp: u64,
    /// Whether this entry has been made visible to other threads.
    pub propagated: bool,
}

/// A per-thread write buffer for ARM/RISC-V models.
///
/// Unlike the FIFO store buffer, entries can be drained out of order
/// (modeling the relaxed propagation of ARM/RISC-V).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WriteBuffer {
    pub thread_id: ThreadId,
    pub entries: Vec<WriteBufferEntry>,
    next_timestamp: u64,
}

impl WriteBuffer {
    pub fn new(thread_id: ThreadId) -> Self {
        Self {
            thread_id,
            entries: Vec::new(),
            next_timestamp: 0,
        }
    }

    /// Add a write to the buffer.
    pub fn push(&mut self, address: Address, value: Value, scope: Scope) {
        self.entries.push(WriteBufferEntry {
            address,
            value,
            scope,
            timestamp: self.next_timestamp,
            propagated: false,
        });
        self.next_timestamp += 1;
    }

    /// Propagate (drain) any single unpropagated entry (non-deterministic choice).
    /// Returns the entry that was propagated, if any.
    pub fn propagate_any(&mut self) -> Option<WriteBufferEntry> {
        if let Some(idx) = self.entries.iter().position(|e| !e.propagated) {
            let entry = self.entries.remove(idx);
            Some(entry)
        } else {
            None
        }
    }

    /// Propagate all entries for a given address (fence behavior).
    pub fn propagate_address(&mut self, address: Address) -> Vec<WriteBufferEntry> {
        let mut propagated = Vec::new();
        self.entries.retain(|e| {
            if e.address == address && !e.propagated {
                propagated.push(e.clone());
                false
            } else {
                true
            }
        });
        propagated
    }

    /// Propagate all entries (full fence).
    pub fn propagate_all(&mut self) -> Vec<WriteBufferEntry> {
        let all = self.entries.drain(..).collect();
        all
    }

    /// Look up the most recent buffered write to a given address.
    pub fn lookup(&self, address: Address) -> Option<Value> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.address == address)
            .map(|e| e.value)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ScopeHierarchy — GPU scope hierarchy for operational model
// ═══════════════════════════════════════════════════════════════════════

/// GPU scope hierarchy for the operational model.
#[derive(Debug, Clone)]
pub struct ScopeHierarchy {
    /// Thread → warp mapping.
    pub thread_to_warp: HashMap<ThreadId, usize>,
    /// Thread → CTA mapping.
    pub thread_to_cta: HashMap<ThreadId, usize>,
    /// Thread → GPU mapping.
    pub thread_to_gpu: HashMap<ThreadId, usize>,
}

impl ScopeHierarchy {
    pub fn new() -> Self {
        Self {
            thread_to_warp: HashMap::new(),
            thread_to_cta: HashMap::new(),
            thread_to_gpu: HashMap::new(),
        }
    }

    pub fn assign(&mut self, thread: ThreadId, warp: usize, cta: usize, gpu: usize) {
        self.thread_to_warp.insert(thread, warp);
        self.thread_to_cta.insert(thread, cta);
        self.thread_to_gpu.insert(thread, gpu);
    }

    /// Check if two threads are in the same scope at the given level.
    pub fn same_scope(&self, t1: ThreadId, t2: ThreadId, scope: Scope) -> bool {
        match scope {
            Scope::CTA => self.thread_to_cta.get(&t1) == self.thread_to_cta.get(&t2),
            Scope::GPU => self.thread_to_gpu.get(&t1) == self.thread_to_gpu.get(&t2),
            Scope::System => true,
            Scope::None => t1 == t2,
        }
    }

    /// Get the minimal scope that contains both threads.
    pub fn minimal_common_scope(&self, t1: ThreadId, t2: ThreadId) -> Scope {
        if t1 == t2 {
            return Scope::None;
        }
        if self.thread_to_warp.get(&t1) == self.thread_to_warp.get(&t2) {
            return Scope::None;
        }
        if self.thread_to_cta.get(&t1) == self.thread_to_cta.get(&t2) {
            return Scope::CTA;
        }
        if self.thread_to_gpu.get(&t1) == self.thread_to_gpu.get(&t2) {
            return Scope::GPU;
        }
        Scope::System
    }
}

impl Default for ScopeHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MachineState — global machine state
// ═══════════════════════════════════════════════════════════════════════

/// Global machine state for the operational model.
#[derive(Debug, Clone)]
pub struct MachineState {
    /// Per-thread states.
    pub threads: Vec<ThreadState>,
    /// Shared (global) memory.
    pub shared_memory: HashMap<Address, Value>,
    /// Store buffers (TSO/PSO).
    pub store_buffers: Vec<StoreBuffer>,
    /// Write buffers (ARM/RISC-V/GPU).
    pub write_buffers: Vec<WriteBuffer>,
    /// Memory model kind.
    pub model_kind: MemoryModelKind,
    /// GPU scope hierarchy (if GPU model).
    pub scope_hierarchy: Option<ScopeHierarchy>,
    /// Global timestamp counter.
    pub global_timestamp: u64,
    /// Execution trace (events generated so far).
    pub trace: Vec<TraceEvent>,
}

/// A single event in the execution trace.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub event_id: usize,
    pub thread_id: ThreadId,
    pub op_type: OpType,
    pub address: Address,
    pub value: Value,
    pub timestamp: u64,
}

impl MachineState {
    /// Create a new machine state for the given model.
    pub fn new(num_threads: usize, model_kind: MemoryModelKind) -> Self {
        let threads = (0..num_threads)
            .map(|i| ThreadState::new(i))
            .collect();
        let store_buffers = (0..num_threads)
            .map(|i| StoreBuffer::new(i))
            .collect();
        let write_buffers = (0..num_threads)
            .map(|i| WriteBuffer::new(i))
            .collect();

        Self {
            threads,
            shared_memory: HashMap::new(),
            store_buffers,
            write_buffers,
            model_kind,
            scope_hierarchy: None,
            global_timestamp: 0,
            trace: Vec::new(),
        }
    }

    pub fn with_scope_hierarchy(mut self, hierarchy: ScopeHierarchy) -> Self {
        self.scope_hierarchy = Some(hierarchy);
        self
    }

    /// Initialize shared memory.
    pub fn init_memory(&mut self, address: Address, value: Value) {
        self.shared_memory.insert(address, value);
    }

    /// Read a value from memory (respecting the memory model).
    pub fn read(&self, thread: ThreadId, address: Address) -> Value {
        match self.model_kind {
            MemoryModelKind::SC => {
                // SC: read directly from shared memory
                self.shared_memory.get(&address).copied().unwrap_or(0)
            }
            MemoryModelKind::TSO | MemoryModelKind::PSO => {
                // TSO/PSO: check store buffer first
                if let Some(val) = self.store_buffers[thread].lookup(address) {
                    val
                } else {
                    self.shared_memory.get(&address).copied().unwrap_or(0)
                }
            }
            MemoryModelKind::ARM | MemoryModelKind::RISCV | MemoryModelKind::GPU => {
                // ARM/RISC-V/GPU: check write buffer first
                if let Some(val) = self.write_buffers[thread].lookup(address) {
                    val
                } else {
                    self.shared_memory.get(&address).copied().unwrap_or(0)
                }
            }
        }
    }

    /// Write a value to memory (respecting the memory model).
    pub fn write(&mut self, thread: ThreadId, address: Address, value: Value) {
        match self.model_kind {
            MemoryModelKind::SC => {
                // SC: write directly to shared memory
                self.shared_memory.insert(address, value);
            }
            MemoryModelKind::TSO | MemoryModelKind::PSO => {
                // TSO/PSO: write to store buffer
                self.store_buffers[thread].push(address, value);
            }
            MemoryModelKind::ARM | MemoryModelKind::RISCV | MemoryModelKind::GPU => {
                // ARM/RISC-V/GPU: write to write buffer
                let scope = self.threads[thread].scope;
                self.write_buffers[thread].push(address, value, scope);
            }
        }

        // Record trace event
        let event_id = self.trace.len();
        self.trace.push(TraceEvent {
            event_id,
            thread_id: thread,
            op_type: OpType::Write,
            address,
            value,
            timestamp: self.global_timestamp,
        });
        self.global_timestamp += 1;
    }

    /// Execute a fence on the given thread.
    pub fn fence(&mut self, thread: ThreadId) {
        match self.model_kind {
            MemoryModelKind::SC => {
                // No-op for SC
            }
            MemoryModelKind::TSO | MemoryModelKind::PSO => {
                // Flush store buffer
                let entries = self.store_buffers[thread].flush_all();
                for entry in entries {
                    self.shared_memory.insert(entry.address, entry.value);
                }
            }
            MemoryModelKind::ARM | MemoryModelKind::RISCV | MemoryModelKind::GPU => {
                // Propagate all writes
                let entries = self.write_buffers[thread].propagate_all();
                for entry in entries {
                    self.shared_memory.insert(entry.address, entry.value);
                }
            }
        }

        let event_id = self.trace.len();
        self.trace.push(TraceEvent {
            event_id,
            thread_id: thread,
            op_type: OpType::Fence,
            address: 0,
            value: 0,
            timestamp: self.global_timestamp,
        });
        self.global_timestamp += 1;
    }

    /// Check if the system has terminated.
    pub fn is_terminated(&self) -> bool {
        self.threads.iter().all(|t| t.terminated)
            && self.store_buffers.iter().all(|b| b.is_empty())
            && self.write_buffers.iter().all(|b| b.is_empty())
    }

    /// Get the number of active (non-terminated) threads.
    pub fn active_thread_count(&self) -> usize {
        self.threads.iter().filter(|t| !t.terminated).count()
    }

    /// Get all possible next states (non-deterministic step).
    pub fn step(&self) -> Vec<MachineState> {
        let mut successors = Vec::new();

        // For each active thread, it can either:
        // 1. Execute its next instruction
        // 2. Have a store buffer entry drain (TSO/PSO)
        // 3. Have a write buffer entry propagate (ARM/RISC-V/GPU)

        for tid in 0..self.threads.len() {
            if self.threads[tid].terminated {
                continue;
            }

            // Option 1: Thread executes a read (reads from memory)
            {
                let mut next = self.clone();
                let addr = (tid as u64) * 0x100; // Simplified
                let val = next.read(tid, addr);
                next.threads[tid].write_register(0, val);
                next.threads[tid].advance_pc();
                let event_id = next.trace.len();
                next.trace.push(TraceEvent {
                    event_id,
                    thread_id: tid,
                    op_type: OpType::Read,
                    address: addr,
                    value: val,
                    timestamp: next.global_timestamp,
                });
                next.global_timestamp += 1;
                successors.push(next);
            }
        }

        // Store buffer drains (TSO/PSO)
        if matches!(self.model_kind, MemoryModelKind::TSO | MemoryModelKind::PSO) {
            for tid in 0..self.threads.len() {
                if !self.store_buffers[tid].is_empty() {
                    let mut next = self.clone();
                    if let Some(entry) = next.store_buffers[tid].drain_oldest() {
                        next.shared_memory.insert(entry.address, entry.value);
                    }
                    successors.push(next);
                }
            }
        }

        // Write buffer propagations (ARM/RISC-V/GPU)
        if matches!(
            self.model_kind,
            MemoryModelKind::ARM | MemoryModelKind::RISCV | MemoryModelKind::GPU
        ) {
            for tid in 0..self.threads.len() {
                if !self.write_buffers[tid].is_empty() {
                    let mut next = self.clone();
                    if let Some(entry) = next.write_buffers[tid].propagate_any() {
                        next.shared_memory.insert(entry.address, entry.value);
                    }
                    successors.push(next);
                }
            }
        }

        successors
    }
}

// ═══════════════════════════════════════════════════════════════════════
// OperationalModel — top-level API
// ═══════════════════════════════════════════════════════════════════════

/// The operational memory model engine.
#[derive(Debug, Clone)]
pub struct OperationalModel {
    pub kind: MemoryModelKind,
    pub max_states: usize,
}

impl OperationalModel {
    pub fn new(kind: MemoryModelKind) -> Self {
        Self {
            kind,
            max_states: 10_000,
        }
    }

    pub fn with_max_states(mut self, max: usize) -> Self {
        self.max_states = max;
        self
    }

    /// Enumerate all reachable states from an initial state.
    pub fn enumerate_states(&self, initial: &MachineState) -> ExecutionEnumeration {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut queue: VecDeque<MachineState> = VecDeque::new();
        let mut terminal_states: Vec<MachineState> = Vec::new();
        let mut total_states = 0usize;

        let initial_hash = self.state_hash(initial);
        visited.insert(initial_hash);
        queue.push_back(initial.clone());

        while let Some(state) = queue.pop_front() {
            total_states += 1;
            if total_states > self.max_states {
                return ExecutionEnumeration {
                    terminal_states,
                    total_states_explored: total_states,
                    exhaustive: false,
                };
            }

            if state.is_terminated() {
                terminal_states.push(state);
                continue;
            }

            let successors = state.step();
            for succ in successors {
                let h = self.state_hash(&succ);
                if visited.insert(h) {
                    queue.push_back(succ);
                }
            }
        }

        ExecutionEnumeration {
            terminal_states,
            total_states_explored: total_states,
            exhaustive: true,
        }
    }

    /// Simple hash for state deduplication.
    fn state_hash(&self, state: &MachineState) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for t in &state.threads {
            t.pc.hash(&mut hasher);
            t.terminated.hash(&mut hasher);
            let mut reg_keys: Vec<_> = t.registers.iter().collect();
            reg_keys.sort_by_key(|&(k, _)| *k);
            for (k, v) in reg_keys {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
        let mut mem_keys: Vec<_> = state.shared_memory.iter().collect();
        mem_keys.sort_by_key(|&(k, _)| *k);
        for (k, v) in mem_keys {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        for sb in &state.store_buffers {
            sb.entries.len().hash(&mut hasher);
        }
        for wb in &state.write_buffers {
            wb.entries.len().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Convert the trace from an operational execution to an execution graph.
    pub fn trace_to_graph(trace: &[TraceEvent]) -> ExecutionGraph {
        let events: Vec<Event> = trace
            .iter()
            .enumerate()
            .map(|(idx, te)| {
                Event::new(idx, te.thread_id, te.op_type, te.address, te.value)
                    .with_po_index(idx)
            })
            .collect();

        let n = events.len();
        let mut graph = ExecutionGraph::new(events);

        // Build rf: for each read, find the latest write with matching value
        let mut rf = BitMatrix::new(n);
        for (r_idx, te) in trace.iter().enumerate() {
            if te.op_type == OpType::Read || te.op_type == OpType::RMW {
                for w_idx in (0..r_idx).rev() {
                    let w_te = &trace[w_idx];
                    if (w_te.op_type == OpType::Write || w_te.op_type == OpType::RMW)
                        && w_te.address == te.address
                        && w_te.value == te.value
                    {
                        rf.set(w_idx, r_idx, true);
                        break;
                    }
                }
            }
        }
        graph.rf = rf;

        // Build co: writes to same address ordered by timestamp
        let mut addr_writes: HashMap<Address, Vec<usize>> = HashMap::new();
        for (idx, te) in trace.iter().enumerate() {
            if te.op_type == OpType::Write || te.op_type == OpType::RMW {
                addr_writes.entry(te.address).or_default().push(idx);
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
        graph.fr = graph.rf.inverse().compose(&graph.co);

        graph
    }
}

/// Result of operational execution enumeration.
#[derive(Debug, Clone)]
pub struct ExecutionEnumeration {
    /// Terminal (fully executed) states.
    pub terminal_states: Vec<MachineState>,
    /// Total states explored.
    pub total_states_explored: usize,
    /// Whether the enumeration was exhaustive.
    pub exhaustive: bool,
}

impl ExecutionEnumeration {
    /// Number of distinct final states.
    pub fn num_outcomes(&self) -> usize {
        self.terminal_states.len()
    }

    /// Get all distinct final memory states.
    pub fn final_memories(&self) -> Vec<HashMap<Address, Value>> {
        self.terminal_states
            .iter()
            .map(|s| s.shared_memory.clone())
            .collect()
    }
}

impl fmt::Display for ExecutionEnumeration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Enumeration: {} outcomes from {} states (exhaustive: {})",
            self.num_outcomes(),
            self.total_states_explored,
            self.exhaustive,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// OperationalAxiomaticEquivalence — checking correspondence
// ═══════════════════════════════════════════════════════════════════════

/// Result of checking operational-axiomatic equivalence.
#[derive(Debug, Clone)]
pub struct OpAxEquivalence {
    pub equivalent: bool,
    pub op_outcomes: HashSet<Vec<(Address, Value)>>,
    pub ax_outcomes: HashSet<Vec<(Address, Value)>>,
    pub only_in_op: HashSet<Vec<(Address, Value)>>,
    pub only_in_ax: HashSet<Vec<(Address, Value)>>,
}

impl OpAxEquivalence {
    /// Check equivalence: every outcome allowed operationally should be
    /// allowed axiomatically, and vice versa.
    pub fn check(
        op_enum: &ExecutionEnumeration,
        ax_graphs: &[ExecutionGraph],
        ax_consistent: &[bool],
    ) -> Self {
        let op_outcomes: HashSet<Vec<(Address, Value)>> = op_enum
            .terminal_states
            .iter()
            .map(|s| {
                let mut mem: Vec<(Address, Value)> = s.shared_memory.iter()
                    .map(|(&a, &v)| (a, v))
                    .collect();
                mem.sort();
                mem
            })
            .collect();

        let ax_outcomes: HashSet<Vec<(Address, Value)>> = ax_graphs
            .iter()
            .zip(ax_consistent.iter())
            .filter(|(_, &consistent)| consistent)
            .map(|(g, _)| {
                let mut mem: Vec<(Address, Value)> = Vec::new();
                for event in &g.events {
                    if event.op_type == OpType::Write || event.op_type == OpType::RMW {
                        mem.push((event.address, event.value));
                    }
                }
                mem.sort();
                mem.dedup();
                mem
            })
            .collect();

        let only_in_op: HashSet<_> = op_outcomes.difference(&ax_outcomes).cloned().collect();
        let only_in_ax: HashSet<_> = ax_outcomes.difference(&op_outcomes).cloned().collect();

        OpAxEquivalence {
            equivalent: only_in_op.is_empty() && only_in_ax.is_empty(),
            op_outcomes,
            ax_outcomes,
            only_in_op,
            only_in_ax,
        }
    }
}

impl fmt::Display for OpAxEquivalence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Op-Ax equivalence: {} (op: {} outcomes, ax: {} outcomes, only_op: {}, only_ax: {})",
            if self.equivalent { "EQUIVALENT" } else { "NOT EQUIVALENT" },
            self.op_outcomes.len(),
            self.ax_outcomes.len(),
            self.only_in_op.len(),
            self.only_in_ax.len(),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // --- ThreadState tests ---

    #[test]
    fn test_thread_state_creation() {
        let ts = ThreadState::new(0);
        assert_eq!(ts.thread_id, 0);
        assert_eq!(ts.pc, 0);
        assert!(!ts.terminated);
    }

    #[test]
    fn test_thread_state_registers() {
        let mut ts = ThreadState::new(0);
        assert_eq!(ts.read_register(0), 0);
        ts.write_register(0, 42);
        assert_eq!(ts.read_register(0), 42);
    }

    #[test]
    fn test_thread_state_pc() {
        let mut ts = ThreadState::new(0);
        ts.advance_pc();
        assert_eq!(ts.pc, 1);
        ts.set_pc(10);
        assert_eq!(ts.pc, 10);
    }

    #[test]
    fn test_thread_state_terminate() {
        let mut ts = ThreadState::new(0);
        assert!(!ts.terminated);
        ts.terminate();
        assert!(ts.terminated);
    }

    // --- StoreBuffer tests ---

    #[test]
    fn test_store_buffer_empty() {
        let sb = StoreBuffer::new(0);
        assert!(sb.is_empty());
        assert_eq!(sb.len(), 0);
        assert_eq!(sb.lookup(0x100), None);
    }

    #[test]
    fn test_store_buffer_push_lookup() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 42);
        assert!(!sb.is_empty());
        assert_eq!(sb.len(), 1);
        assert_eq!(sb.lookup(0x100), Some(42));
        assert_eq!(sb.lookup(0x200), None);
    }

    #[test]
    fn test_store_buffer_fifo() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        sb.push(0x100, 3);

        // Lookup returns most recent
        assert_eq!(sb.lookup(0x100), Some(3));

        // Drain returns oldest
        let entry = sb.drain_oldest().unwrap();
        assert_eq!(entry.address, 0x100);
        assert_eq!(entry.value, 1);
    }

    #[test]
    fn test_store_buffer_flush_address() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        sb.push(0x100, 3);

        let flushed = sb.flush_address(0x100);
        assert_eq!(flushed.len(), 2);
        assert_eq!(sb.len(), 1);
        assert_eq!(sb.lookup(0x200), Some(2));
    }

    #[test]
    fn test_store_buffer_flush_all() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);

        let flushed = sb.flush_all();
        assert_eq!(flushed.len(), 2);
        assert!(sb.is_empty());
    }

    // --- WriteBuffer tests ---

    #[test]
    fn test_write_buffer_empty() {
        let wb = WriteBuffer::new(0);
        assert!(wb.is_empty());
        assert_eq!(wb.len(), 0);
    }

    #[test]
    fn test_write_buffer_push_lookup() {
        let mut wb = WriteBuffer::new(0);
        wb.push(0x100, 42, Scope::None);
        assert_eq!(wb.lookup(0x100), Some(42));
        assert_eq!(wb.lookup(0x200), None);
    }

    #[test]
    fn test_write_buffer_propagate_any() {
        let mut wb = WriteBuffer::new(0);
        wb.push(0x100, 1, Scope::None);
        wb.push(0x200, 2, Scope::None);

        let entry = wb.propagate_any().unwrap();
        assert!(entry.address == 0x100 || entry.address == 0x200);
        assert_eq!(wb.len(), 1);
    }

    #[test]
    fn test_write_buffer_propagate_all() {
        let mut wb = WriteBuffer::new(0);
        wb.push(0x100, 1, Scope::None);
        wb.push(0x200, 2, Scope::CTA);

        let entries = wb.propagate_all();
        assert_eq!(entries.len(), 2);
        assert!(wb.is_empty());
    }

    // --- ScopeHierarchy tests ---

    #[test]
    fn test_scope_hierarchy() {
        let mut sh = ScopeHierarchy::new();
        sh.assign(0, 0, 0, 0);
        sh.assign(1, 0, 0, 0); // Same warp, CTA, GPU as thread 0
        sh.assign(2, 1, 0, 0); // Different warp, same CTA
        sh.assign(3, 2, 1, 0); // Different CTA, same GPU

        assert!(sh.same_scope(0, 1, Scope::CTA));
        assert!(sh.same_scope(0, 2, Scope::CTA));
        assert!(!sh.same_scope(0, 3, Scope::CTA));
        assert!(sh.same_scope(0, 3, Scope::GPU));
        assert!(sh.same_scope(0, 3, Scope::System));
    }

    #[test]
    fn test_scope_hierarchy_minimal_common() {
        let mut sh = ScopeHierarchy::new();
        sh.assign(0, 0, 0, 0);
        sh.assign(1, 0, 0, 0);
        sh.assign(2, 1, 0, 0);
        sh.assign(3, 2, 1, 0);
        sh.assign(4, 3, 2, 1);

        assert_eq!(sh.minimal_common_scope(0, 0), Scope::None);
        assert_eq!(sh.minimal_common_scope(0, 1), Scope::None); // Same warp
        assert_eq!(sh.minimal_common_scope(0, 2), Scope::CTA); // Different warp, same CTA
        assert_eq!(sh.minimal_common_scope(0, 3), Scope::GPU); // Different CTA, same GPU
        assert_eq!(sh.minimal_common_scope(0, 4), Scope::System); // Different GPU
    }

    // --- MachineState tests ---

    #[test]
    fn test_machine_state_sc() {
        let mut state = MachineState::new(2, MemoryModelKind::SC);
        state.init_memory(0x100, 0);

        // SC: writes go directly to memory
        state.write(0, 0x100, 42);
        assert_eq!(state.read(1, 0x100), 42);
    }

    #[test]
    fn test_machine_state_tso() {
        let mut state = MachineState::new(2, MemoryModelKind::TSO);
        state.init_memory(0x100, 0);

        // TSO: writes go to store buffer
        state.write(0, 0x100, 42);

        // Same thread sees the write (store forwarding)
        assert_eq!(state.read(0, 0x100), 42);

        // Different thread sees old value
        assert_eq!(state.read(1, 0x100), 0);

        // After fence, different thread sees new value
        state.fence(0);
        assert_eq!(state.read(1, 0x100), 42);
    }

    #[test]
    fn test_machine_state_arm() {
        let mut state = MachineState::new(2, MemoryModelKind::ARM);
        state.init_memory(0x100, 0);

        // ARM: writes go to write buffer
        state.write(0, 0x100, 42);

        // Same thread sees the write
        assert_eq!(state.read(0, 0x100), 42);

        // Different thread sees old value
        assert_eq!(state.read(1, 0x100), 0);

        // Fence propagates all writes
        state.fence(0);
        assert_eq!(state.read(1, 0x100), 42);
    }

    #[test]
    fn test_machine_state_terminated() {
        let mut state = MachineState::new(2, MemoryModelKind::SC);
        assert!(!state.is_terminated());
        assert_eq!(state.active_thread_count(), 2);

        state.threads[0].terminate();
        state.threads[1].terminate();
        assert!(state.is_terminated());
        assert_eq!(state.active_thread_count(), 0);
    }

    #[test]
    fn test_machine_state_trace() {
        let mut state = MachineState::new(2, MemoryModelKind::SC);
        state.write(0, 0x100, 1);
        state.write(1, 0x200, 2);
        state.fence(0);

        assert_eq!(state.trace.len(), 3);
        assert_eq!(state.trace[0].op_type, OpType::Write);
        assert_eq!(state.trace[2].op_type, OpType::Fence);
    }

    // --- OperationalModel tests ---

    #[test]
    fn test_operational_model_creation() {
        let model = OperationalModel::new(MemoryModelKind::TSO);
        assert_eq!(model.kind, MemoryModelKind::TSO);
    }

    #[test]
    fn test_trace_to_graph() {
        let trace = vec![
            TraceEvent {
                event_id: 0,
                thread_id: 0,
                op_type: OpType::Write,
                address: 0x100,
                value: 1,
                timestamp: 0,
            },
            TraceEvent {
                event_id: 1,
                thread_id: 1,
                op_type: OpType::Read,
                address: 0x100,
                value: 1,
                timestamp: 1,
            },
        ];

        let graph = OperationalModel::trace_to_graph(&trace);
        assert_eq!(graph.events.len(), 2);
        assert!(graph.rf.get(0, 1)); // Write reads-from Read
    }

    #[test]
    fn test_trace_to_graph_coherence() {
        let trace = vec![
            TraceEvent {
                event_id: 0,
                thread_id: 0,
                op_type: OpType::Write,
                address: 0x100,
                value: 1,
                timestamp: 0,
            },
            TraceEvent {
                event_id: 1,
                thread_id: 0,
                op_type: OpType::Write,
                address: 0x100,
                value: 2,
                timestamp: 1,
            },
        ];

        let graph = OperationalModel::trace_to_graph(&trace);
        assert!(graph.co.get(0, 1));
        assert!(!graph.co.get(1, 0));
    }

    // --- Step function tests ---

    #[test]
    fn test_machine_state_step() {
        let mut state = MachineState::new(2, MemoryModelKind::SC);
        state.init_memory(0, 0);
        let successors = state.step();
        assert!(!successors.is_empty());
    }

    #[test]
    fn test_machine_state_step_tso_drain() {
        let mut state = MachineState::new(1, MemoryModelKind::TSO);
        state.store_buffers[0].push(0x100, 42);

        let successors = state.step();
        // Should include a drain transition
        assert!(successors.len() >= 1);
    }

    // --- MemoryModelKind tests ---

    #[test]
    fn test_model_kind_display() {
        assert_eq!(format!("{}", MemoryModelKind::SC), "SC");
        assert_eq!(format!("{}", MemoryModelKind::TSO), "TSO");
        assert_eq!(format!("{}", MemoryModelKind::ARM), "ARM");
        assert_eq!(format!("{}", MemoryModelKind::GPU), "GPU");
    }

    // --- OpAxEquivalence tests ---

    #[test]
    fn test_op_ax_equivalence_trivial() {
        let enum_result = ExecutionEnumeration {
            terminal_states: Vec::new(),
            total_states_explored: 0,
            exhaustive: true,
        };

        let eq = OpAxEquivalence::check(&enum_result, &[], &[]);
        assert!(eq.equivalent);
    }

    // --- ExecutionEnumeration tests ---

    #[test]
    fn test_execution_enumeration_display() {
        let enum_result = ExecutionEnumeration {
            terminal_states: Vec::new(),
            total_states_explored: 100,
            exhaustive: true,
        };
        let s = format!("{}", enum_result);
        assert!(s.contains("0 outcomes"));
        assert!(s.contains("100 states"));
    }

    #[test]
    fn test_execution_enumeration_final_memories() {
        let mut state = MachineState::new(1, MemoryModelKind::SC);
        state.shared_memory.insert(0x100, 42);
        state.threads[0].terminate();

        let enum_result = ExecutionEnumeration {
            terminal_states: vec![state],
            total_states_explored: 1,
            exhaustive: true,
        };

        let mems = enum_result.final_memories();
        assert_eq!(mems.len(), 1);
        assert_eq!(mems[0].get(&0x100), Some(&42));
    }

    // --- Store buffer edge cases ---

    #[test]
    fn test_store_buffer_max_size() {
        let mut sb = StoreBuffer::new(0).with_max_size(2);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        assert!(sb.is_full());
    }
}
