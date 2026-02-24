//! x86-TSO (Total Store Order) memory model for the LITMUS∞ checker.
//!
//! Implements the x86-TSO memory model including store buffer modeling,
//! MFENCE semantics, LOCK prefix handling, TSO-consistent execution
//! checking, and operational semantics with store buffer nondeterminism.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, EventId, ThreadId, Address, Value, OpType, Scope,
    ExecutionGraph, ExecutionGraphBuilder, BitMatrix,
};
use crate::checker::memory_model::{
    MemoryModel, RelationExpr, PredicateExpr,
};
use crate::checker::litmus::{LitmusTest, Thread, Instruction, Ordering};

// ═══════════════════════════════════════════════════════════════════════════
// Store Buffer
// ═══════════════════════════════════════════════════════════════════════════

/// An entry in a thread's store buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreBufferEntry {
    /// Memory address.
    pub address: Address,
    /// Value to be written.
    pub value: Value,
    /// Logical timestamp (for ordering).
    pub timestamp: u64,
    /// Whether this entry has been flushed to memory.
    pub flushed: bool,
}

/// Per-thread FIFO store buffer (models x86 write combining).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreBuffer {
    /// FIFO queue of pending stores.
    entries: VecDeque<StoreBufferEntry>,
    /// Thread ID.
    pub thread_id: ThreadId,
    /// Next timestamp to assign.
    next_timestamp: u64,
}

impl StoreBuffer {
    /// Create a new empty store buffer.
    pub fn new(thread_id: ThreadId) -> Self {
        StoreBuffer {
            entries: VecDeque::new(),
            thread_id,
            next_timestamp: 0,
        }
    }

    /// Push a new store into the buffer.
    pub fn push(&mut self, address: Address, value: Value) {
        let ts = self.next_timestamp;
        self.next_timestamp += 1;
        self.entries.push_back(StoreBufferEntry {
            address,
            value,
            timestamp: ts,
            flushed: false,
        });
    }

    /// Read from the store buffer: return the most recent unflushed value
    /// for the given address, or None if no buffered store to that address.
    pub fn read_from_buffer(&self, address: Address) -> Option<Value> {
        self.entries.iter().rev()
            .find(|e| e.address == address && !e.flushed)
            .map(|e| e.value)
    }

    /// Flush the oldest unflushed entry. Returns the flushed entry if any.
    pub fn flush_oldest(&mut self) -> Option<StoreBufferEntry> {
        for entry in self.entries.iter_mut() {
            if !entry.flushed {
                entry.flushed = true;
                return Some(entry.clone());
            }
        }
        None
    }

    /// Flush all entries (used for MFENCE / LOCK).
    pub fn flush_all(&mut self) -> Vec<StoreBufferEntry> {
        let mut flushed = Vec::new();
        for entry in self.entries.iter_mut() {
            if !entry.flushed {
                entry.flushed = true;
                flushed.push(entry.clone());
            }
        }
        flushed
    }

    /// Whether the buffer has no unflushed entries.
    pub fn is_empty(&self) -> bool {
        !self.entries.iter().any(|e| !e.flushed)
    }

    /// Number of unflushed entries.
    pub fn pending_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.flushed).count()
    }

    /// Pending entries for a specific address.
    pub fn pending_for_address(&self, address: Address) -> Vec<&StoreBufferEntry> {
        self.entries.iter()
            .filter(|e| !e.flushed && e.address == address)
            .collect()
    }

    /// Clean up flushed entries from the front.
    pub fn gc(&mut self) {
        while self.entries.front().map_or(false, |e| e.flushed) {
            self.entries.pop_front();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TSO State
// ═══════════════════════════════════════════════════════════════════════════

/// Global state of a TSO machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TSOState {
    /// Shared memory.
    pub memory: BTreeMap<Address, Value>,
    /// Per-thread store buffers.
    pub buffers: Vec<StoreBuffer>,
    /// Per-thread program counters.
    pub pc: Vec<usize>,
    /// Per-thread register files.
    pub registers: Vec<BTreeMap<usize, Value>>,
    /// Number of threads.
    pub num_threads: usize,
}

impl TSOState {
    /// Create initial state with given number of threads.
    pub fn new(num_threads: usize) -> Self {
        TSOState {
            memory: BTreeMap::new(),
            buffers: (0..num_threads).map(|t| StoreBuffer::new(t)).collect(),
            pc: vec![0; num_threads],
            registers: vec![BTreeMap::new(); num_threads],
            num_threads,
        }
    }

    /// Set initial memory value.
    pub fn set_initial(&mut self, addr: Address, val: Value) {
        self.memory.insert(addr, val);
    }

    /// Read: check thread's store buffer first, then shared memory.
    pub fn read(&self, thread: ThreadId, addr: Address) -> Value {
        // Check store buffer first (store-buffer forwarding)
        if let Some(val) = self.buffers[thread].read_from_buffer(addr) {
            return val;
        }
        // Fall back to shared memory
        self.memory.get(&addr).copied().unwrap_or(0)
    }

    /// Write: push to thread's store buffer (not directly to memory).
    pub fn write(&mut self, thread: ThreadId, addr: Address, val: Value) {
        self.buffers[thread].push(addr, val);
    }

    /// MFENCE: flush the thread's entire store buffer to memory.
    pub fn fence(&mut self, thread: ThreadId) {
        let flushed = self.buffers[thread].flush_all();
        for entry in flushed {
            self.memory.insert(entry.address, entry.value);
        }
    }

    /// LOCK: acquire lock (flush buffer), perform operation, release lock (flush buffer).
    pub fn lock_op(&mut self, thread: ThreadId, addr: Address, val: Value) -> Value {
        // Flush first
        self.fence(thread);
        // Read current value
        let old = self.memory.get(&addr).copied().unwrap_or(0);
        // Write new value directly to memory
        self.memory.insert(addr, val);
        old
    }

    /// Nondeterministically flush one pending store from any thread.
    /// Returns all possible next states.
    pub fn nondeterministic_flush(&self) -> Vec<TSOState> {
        let mut successors = Vec::new();
        for t in 0..self.num_threads {
            if !self.buffers[t].is_empty() {
                let mut next = self.clone();
                if let Some(entry) = next.buffers[t].flush_oldest() {
                    next.memory.insert(entry.address, entry.value);
                    next.buffers[t].gc();
                    successors.push(next);
                }
            }
        }
        successors
    }

    /// Check if all store buffers are empty.
    pub fn all_buffers_empty(&self) -> bool {
        self.buffers.iter().all(|b| b.is_empty())
    }

    /// Get register value.
    pub fn get_reg(&self, thread: ThreadId, reg: usize) -> Value {
        self.registers[thread].get(&reg).copied().unwrap_or(0)
    }

    /// Set register value.
    pub fn set_reg(&mut self, thread: ThreadId, reg: usize, val: Value) {
        self.registers[thread].insert(reg, val);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TSO Transitions
// ═══════════════════════════════════════════════════════════════════════════

/// A single transition in the TSO operational semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TSOTransition {
    /// Thread performs a load.
    Read { thread: ThreadId, addr: Address, reg: usize, value: Value },
    /// Thread performs a store (into store buffer).
    Write { thread: ThreadId, addr: Address, value: Value },
    /// Thread performs a fence (flush store buffer).
    Fence { thread: ThreadId },
    /// Store buffer flush (nondeterministic).
    BufferFlush { thread: ThreadId, addr: Address, value: Value },
    /// LOCK operation.
    Lock { thread: ThreadId, addr: Address, new_value: Value, old_value: Value },
}

impl fmt::Display for TSOTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TSOTransition::Read { thread, addr, reg, value } =>
                write!(f, "T{}:R(r{}={},@{:#x})", thread, reg, value, addr),
            TSOTransition::Write { thread, addr, value } =>
                write!(f, "T{}:W(@{:#x}={})", thread, addr, value),
            TSOTransition::Fence { thread } =>
                write!(f, "T{}:MFENCE", thread),
            TSOTransition::BufferFlush { thread, addr, value } =>
                write!(f, "T{}:Flush(@{:#x}={})", thread, addr, value),
            TSOTransition::Lock { thread, addr, new_value, old_value } =>
                write!(f, "T{}:LOCK(@{:#x}:{}→{})", thread, addr, old_value, new_value),
        }
    }
}

/// An execution trace in the TSO model.
#[derive(Debug, Clone)]
pub struct TSOExecutionTrace {
    /// Sequence of transitions.
    pub transitions: Vec<TSOTransition>,
    /// Initial state.
    pub initial_state: TSOState,
    /// Final state.
    pub final_state: TSOState,
}

impl TSOExecutionTrace {
    /// Create a new trace.
    pub fn new(initial: TSOState) -> Self {
        TSOExecutionTrace {
            transitions: Vec::new(),
            initial_state: initial.clone(),
            final_state: initial,
        }
    }

    /// Add a transition.
    pub fn add(&mut self, transition: TSOTransition, state: TSOState) {
        self.transitions.push(transition);
        self.final_state = state;
    }

    /// Length of the trace.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Is the trace empty?
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Get the final memory state.
    pub fn final_memory(&self) -> &BTreeMap<Address, Value> {
        &self.final_state.memory
    }

    /// Get final register values.
    pub fn final_registers(&self) -> &[BTreeMap<usize, Value>] {
        &self.final_state.registers
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TSO Axioms (Axiomatic Definition)
// ═══════════════════════════════════════════════════════════════════════════

/// x86-TSO axioms in the axiomatic framework.
#[derive(Debug, Clone)]
pub struct TSOAxioms {
    /// Preserved program order under TSO.
    pub ppo_description: String,
    /// Global happens-before construction.
    pub ghb_description: String,
}

impl TSOAxioms {
    /// Build TSO axioms.
    pub fn new() -> Self {
        TSOAxioms {
            ppo_description: "po \\ (W→R on different addresses)".to_string(),
            ghb_description: "ppo ∪ rfe ∪ co ∪ fr".to_string(),
        }
    }

    /// Build the preserved program order relation.
    /// TSO preserves all po except W→R to different addresses.
    pub fn compute_ppo(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut ppo = BitMatrix::new(n);

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.events[i];
                let ej = &exec.events[j];

                // Same thread, i before j in program order
                if ei.thread != ej.thread || ei.po_index >= ej.po_index {
                    continue;
                }

                // TSO allows W→R reordering (to different addresses only)
                let is_wr = ei.op_type == OpType::Write && ej.op_type == OpType::Read;
                let same_addr = ei.address == ej.address;

                if is_wr && !same_addr {
                    // This is the only reordering TSO allows — skip it
                    continue;
                }

                ppo.set(i, j, true);
            }
        }

        ppo
    }

    /// Build the global happens-before relation.
    /// ghb = ppo ∪ rfe ∪ co ∪ fr
    pub fn compute_ghb(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let ppo = Self::compute_ppo(exec);
        let mut ghb = ppo;

        // Add rfe (external reads-from: rf between different threads)
        for (w, r) in exec.rf.edges() {
            if exec.events[w].thread != exec.events[r].thread {
                ghb.set(w, r, true);
            }
        }

        // Add co (coherence order)
        for (w1, w2) in exec.co.edges() {
            ghb.set(w1, w2, true);
        }

        // Add fr (from-reads)
        // fr = rf⁻¹ ; co
        for (w_rf, r) in exec.rf.edges() {
            for (w_co_from, w_co_to) in exec.co.edges() {
                if w_rf == w_co_from {
                    ghb.set(r, w_co_to, true);
                }
            }
        }

        ghb
    }

    /// Check TSO consistency: ghb must be acyclic.
    pub fn check_consistency(exec: &ExecutionGraph) -> bool {
        let ghb = Self::compute_ghb(exec);
        ghb.is_acyclic()
    }

    /// Build MFENCE-augmented ppo: add fence ordering.
    pub fn compute_ppo_with_fences(exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.events.len();
        let mut ppo = Self::compute_ppo(exec);

        // MFENCE: orders all operations before and after it
        for f in 0..n {
            if exec.events[f].op_type != OpType::Fence { continue; }
            let thread = exec.events[f].thread;

            for i in 0..n {
                if exec.events[i].thread != thread { continue; }
                if exec.events[i].po_index >= exec.events[f].po_index { continue; }

                for j in 0..n {
                    if exec.events[j].thread != thread { continue; }
                    if exec.events[j].po_index <= exec.events[f].po_index { continue; }

                    // The fence orders i before j (even W→R)
                    ppo.set(i, j, true);
                }
            }
        }

        ppo
    }
}

impl Default for TSOAxioms {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// X86 Model Checker
// ═══════════════════════════════════════════════════════════════════════════

/// The x86-TSO memory model checker.
#[derive(Debug)]
pub struct X86ModelChecker {
    /// Statistics.
    pub stats: X86CheckStats,
}

/// Statistics from x86 model checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct X86CheckStats {
    /// Number of executions checked.
    pub executions_checked: usize,
    /// Number of consistent executions found.
    pub consistent_count: usize,
    /// Number of inconsistent executions found.
    pub inconsistent_count: usize,
}

impl X86ModelChecker {
    /// Create a new x86 model checker.
    pub fn new() -> Self {
        X86ModelChecker { stats: X86CheckStats::default() }
    }

    /// Build x86-TSO as a `MemoryModel`.
    pub fn build_memory_model() -> MemoryModel {
        let mut model = MemoryModel::new("x86-TSO");

        // ppo: program order minus W→R to different addresses
        model.add_derived(
            "ppo",
            RelationExpr::diff(
                RelationExpr::base("po"),
                RelationExpr::inter(
                    RelationExpr::seq(
                        RelationExpr::filter(PredicateExpr::IsWrite),
                        RelationExpr::filter(PredicateExpr::IsRead),
                    ),
                    RelationExpr::diff(
                        RelationExpr::Identity,
                        RelationExpr::base("loc"),
                    ),
                ),
            ),
            "Preserved program order (TSO)",
        );

        // rfe: external reads-from
        model.add_derived(
            "rfe",
            RelationExpr::diff(
                RelationExpr::base("rf"),
                RelationExpr::base("int"),
            ),
            "External reads-from",
        );

        // fr: from-reads (rf⁻¹;co)
        model.add_derived(
            "fr",
            RelationExpr::seq(
                RelationExpr::inverse(RelationExpr::base("rf")),
                RelationExpr::base("co"),
            ),
            "From-reads relation",
        );

        // ghb: ppo ∪ rfe ∪ co ∪ fr
        model.add_derived(
            "ghb",
            RelationExpr::union_many(vec![
                RelationExpr::base("ppo"),
                RelationExpr::base("rfe"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "Global happens-before",
        );

        model.add_acyclic(RelationExpr::base("ghb"));

        model
    }

    /// Check an execution graph for TSO consistency.
    pub fn check_execution(&mut self, exec: &ExecutionGraph) -> bool {
        self.stats.executions_checked += 1;
        let consistent = TSOAxioms::check_consistency(exec);
        if consistent {
            self.stats.consistent_count += 1;
        } else {
            self.stats.inconsistent_count += 1;
        }
        consistent
    }

    /// Check with MFENCE support.
    pub fn check_execution_with_fences(&mut self, exec: &ExecutionGraph) -> bool {
        self.stats.executions_checked += 1;
        let ppo = TSOAxioms::compute_ppo_with_fences(exec);
        let n = exec.events.len();
        let mut ghb = ppo;

        // Add rfe, co, fr
        for (w, r) in exec.rf.edges() {
            if exec.events[w].thread != exec.events[r].thread {
                ghb.set(w, r, true);
            }
        }
        for (w1, w2) in exec.co.edges() {
            ghb.set(w1, w2, true);
        }
        for (w_rf, r) in exec.rf.edges() {
            for (w_co_from, w_co_to) in exec.co.edges() {
                if w_rf == w_co_from {
                    ghb.set(r, w_co_to, true);
                }
            }
        }

        let consistent = ghb.is_acyclic();
        if consistent {
            self.stats.consistent_count += 1;
        } else {
            self.stats.inconsistent_count += 1;
        }
        consistent
    }
}

impl Default for X86ModelChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TSO Operational Checker
// ═══════════════════════════════════════════════════════════════════════════

/// Operational semantics checker for x86-TSO.
/// Explores all interleavings with store buffer nondeterminism.
#[derive(Debug)]
pub struct TSOOperationalChecker {
    /// Maximum states to explore.
    pub max_states: usize,
    /// Statistics.
    pub stats: TSOOpStats,
}

/// Statistics from operational checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TSOOpStats {
    /// Number of states explored.
    pub states_explored: usize,
    /// Number of distinct final states found.
    pub final_states_found: usize,
    /// Whether exploration completed.
    pub completed: bool,
}

impl TSOOperationalChecker {
    /// Create a new checker with a state bound.
    pub fn new(max_states: usize) -> Self {
        TSOOperationalChecker {
            max_states,
            stats: TSOOpStats::default(),
        }
    }

    /// Explore all reachable final states from initial state.
    pub fn explore_all(
        &mut self,
        initial: TSOState,
        programs: &[Vec<TSOInstruction>],
    ) -> Vec<TSOFinalState> {
        let mut visited: HashSet<TSOState> = HashSet::new();
        let mut queue = VecDeque::new();
        let mut final_states = Vec::new();

        queue.push_back(initial.clone());
        visited.insert(initial);

        while let Some(state) = queue.pop_front() {
            if self.stats.states_explored >= self.max_states {
                self.stats.completed = false;
                break;
            }
            self.stats.states_explored += 1;

            let successors = self.successor_states(&state, programs);

            if successors.is_empty() {
                // This is a final state (all threads done, all buffers flushed)
                if state.all_buffers_empty() {
                    let fs = TSOFinalState {
                        memory: state.memory.clone(),
                        registers: state.registers.clone(),
                    };
                    if !final_states.contains(&fs) {
                        final_states.push(fs);
                    }
                } else {
                    // Still need to flush buffers
                    for succ in state.nondeterministic_flush() {
                        if !visited.contains(&succ) {
                            visited.insert(succ.clone());
                            queue.push_back(succ);
                        }
                    }
                }
            } else {
                for succ in successors {
                    if !visited.contains(&succ) {
                        visited.insert(succ.clone());
                        queue.push_back(succ);
                    }
                }
            }
        }

        self.stats.final_states_found = final_states.len();
        self.stats.completed = true;
        final_states
    }

    /// Compute all successor states from the current state.
    fn successor_states(
        &self,
        state: &TSOState,
        programs: &[Vec<TSOInstruction>],
    ) -> Vec<TSOState> {
        let mut successors = Vec::new();

        // Thread transitions
        for t in 0..state.num_threads {
            if state.pc[t] >= programs[t].len() { continue; }
            let instr = &programs[t][state.pc[t]];

            match instr {
                TSOInstruction::Load { reg, addr } => {
                    let mut next = state.clone();
                    let val = next.read(t, *addr);
                    next.set_reg(t, *reg, val);
                    next.pc[t] += 1;
                    successors.push(next);
                }
                TSOInstruction::Store { addr, value } => {
                    let mut next = state.clone();
                    next.write(t, *addr, *value);
                    next.pc[t] += 1;
                    successors.push(next);
                }
                TSOInstruction::MFence => {
                    let mut next = state.clone();
                    next.fence(t);
                    next.pc[t] += 1;
                    successors.push(next);
                }
                TSOInstruction::LockOp { addr, value } => {
                    let mut next = state.clone();
                    let _old = next.lock_op(t, *addr, *value);
                    next.pc[t] += 1;
                    successors.push(next);
                }
            }
        }

        // Nondeterministic buffer flushes
        successors.extend(state.nondeterministic_flush());

        successors
    }
}

/// A TSO instruction for the operational checker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TSOInstruction {
    /// Load from memory.
    Load { reg: usize, addr: Address },
    /// Store to memory.
    Store { addr: Address, value: Value },
    /// Memory fence.
    MFence,
    /// Locked operation.
    LockOp { addr: Address, value: Value },
}

/// A final state observed in TSO exploration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TSOFinalState {
    /// Final memory values.
    pub memory: BTreeMap<Address, Value>,
    /// Final register values per thread.
    pub registers: Vec<BTreeMap<usize, Value>>,
}

impl fmt::Display for TSOFinalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mem: {:?}", self.memory)?;
        for (t, regs) in self.registers.iter().enumerate() {
            write!(f, " | T{}: {:?}", t, regs)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// X86 Litmus Test Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for standard x86 litmus tests.
pub struct X86LitmusBuilder;

impl X86LitmusBuilder {
    /// Store Buffering (SB): the canonical TSO-observable test.
    pub fn store_buffering() -> LitmusTest {
        let mut test = LitmusTest::new("SB");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        t0.load(0, 1, Ordering::Relaxed);   // r0 = y
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed); // y = 1
        t1.load(1, 0, Ordering::Relaxed);   // r1 = x
        test.add_thread(t1);

        test
    }

    /// Store Buffering with MFENCE.
    pub fn store_buffering_fenced() -> LitmusTest {
        let mut test = LitmusTest::new("SB+mfences");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    /// Message Passing (MP).
    pub fn message_passing() -> LitmusTest {
        let mut test = LitmusTest::new("MP");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // data = 1
        t0.store(1, 1, Ordering::Relaxed); // flag = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 1, Ordering::Relaxed);  // r0 = flag
        t1.load(1, 0, Ordering::Relaxed);  // r1 = data
        test.add_thread(t1);

        test
    }

    /// Message Passing with MFENCE.
    pub fn message_passing_fenced() -> LitmusTest {
        let mut test = LitmusTest::new("MP+mfence");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t0.store(1, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 1, Ordering::Relaxed);
        t1.fence(Ordering::SeqCst, crate::checker::litmus::Scope::None);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    /// 2+2W: Two writes by each thread.
    pub fn two_plus_two_writes() -> LitmusTest {
        let mut test = LitmusTest::new("2+2W");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        t0.store(1, 2, Ordering::Relaxed); // y = 2
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed); // y = 1
        t1.store(0, 2, Ordering::Relaxed); // x = 2
        test.add_thread(t1);

        test
    }

    /// Build operational programs from litmus test.
    pub fn to_tso_programs(test: &LitmusTest) -> Vec<Vec<TSOInstruction>> {
        let mut programs = Vec::new();
        for thread in &test.threads {
            let mut prog = Vec::new();
            for instr in &thread.instructions {
                match instr {
                    Instruction::Load { reg, addr, .. } => {
                        prog.push(TSOInstruction::Load { reg: *reg, addr: *addr });
                    }
                    Instruction::Store { addr, value, .. } => {
                        prog.push(TSOInstruction::Store { addr: *addr, value: *value });
                    }
                    Instruction::Fence { .. } => {
                        prog.push(TSOInstruction::MFence);
                    }
                    Instruction::RMW { addr, value, .. } => {
                        prog.push(TSOInstruction::LockOp { addr: *addr, value: *value });
                    }
                    _ => {}
                }
            }
            programs.push(prog);
        }
        programs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TSO Comparison
// ═══════════════════════════════════════════════════════════════════════════

/// Compare TSO vs SC outcomes.
#[derive(Debug)]
pub struct TSOComparison;

impl TSOComparison {
    /// Check if an execution is SC-consistent (all po preserved).
    pub fn is_sc_consistent(exec: &ExecutionGraph) -> bool {
        let n = exec.events.len();
        let mut total_order = BitMatrix::new(n);

        // Under SC, all of po ∪ rf ∪ co ∪ fr must be acyclic
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let ei = &exec.events[i];
                let ej = &exec.events[j];
                if ei.thread == ej.thread && ei.po_index < ej.po_index {
                    total_order.set(i, j, true);
                }
            }
        }
        for (w, r) in exec.rf.edges() {
            total_order.set(w, r, true);
        }
        for (w1, w2) in exec.co.edges() {
            total_order.set(w1, w2, true);
        }
        for (w_rf, r) in exec.rf.edges() {
            for (w_co_from, w_co_to) in exec.co.edges() {
                if w_rf == w_co_from {
                    total_order.set(r, w_co_to, true);
                }
            }
        }

        total_order.is_acyclic()
    }

    /// Check if an execution is TSO-only (TSO-consistent but not SC-consistent).
    pub fn is_tso_only(exec: &ExecutionGraph) -> bool {
        TSOAxioms::check_consistency(exec) && !Self::is_sc_consistent(exec)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_buffer_basic() {
        let mut sb = StoreBuffer::new(0);
        assert!(sb.is_empty());
        sb.push(0x100, 42);
        assert!(!sb.is_empty());
        assert_eq!(sb.pending_count(), 1);
        assert_eq!(sb.read_from_buffer(0x100), Some(42));
        assert_eq!(sb.read_from_buffer(0x200), None);
    }

    #[test]
    fn test_store_buffer_fifo() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x100, 2);
        assert_eq!(sb.read_from_buffer(0x100), Some(2)); // most recent
        let flushed = sb.flush_oldest().unwrap();
        assert_eq!(flushed.value, 1); // oldest first
        assert_eq!(sb.read_from_buffer(0x100), Some(2)); // still 2
    }

    #[test]
    fn test_store_buffer_flush_all() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        sb.push(0x100, 3);
        let flushed = sb.flush_all();
        assert_eq!(flushed.len(), 3);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_tso_state_read_write() {
        let mut state = TSOState::new(2);
        state.set_initial(0x100, 0);
        // Write 42 to x on thread 0
        state.write(0, 0x100, 42);
        // Thread 0 sees the buffered value
        assert_eq!(state.read(0, 0x100), 42);
        // Thread 1 sees the old value (store not flushed)
        assert_eq!(state.read(1, 0x100), 0);
    }

    #[test]
    fn test_tso_state_fence() {
        let mut state = TSOState::new(2);
        state.set_initial(0x100, 0);
        state.write(0, 0x100, 42);
        state.fence(0);
        // After fence, thread 1 sees the value
        assert_eq!(state.read(1, 0x100), 42);
    }

    #[test]
    fn test_tso_nondeterministic_flush() {
        let mut state = TSOState::new(2);
        state.write(0, 0x100, 1);
        state.write(1, 0x200, 2);
        let successors = state.nondeterministic_flush();
        assert_eq!(successors.len(), 2); // can flush from either thread
    }

    #[test]
    fn test_tso_axioms_ppo() {
        let mut builder = ExecutionGraphBuilder::new();
        // Create a simple W→R execution on the same thread
        builder.add_event(0, OpType::Write, 0x100, 1);
        builder.add_event(0, OpType::Read, 0x200, 0);
        let exec = builder.build();
        let ppo = TSOAxioms::compute_ppo(&exec);
        // W→R to different addresses NOT in ppo under TSO
        assert!(!ppo.get(0, 1));
    }

    #[test]
    fn test_tso_axioms_ppo_same_addr() {
        let mut builder = ExecutionGraphBuilder::new();
        builder.add_event(0, OpType::Write, 0x100, 1);
        builder.add_event(0, OpType::Read, 0x100, 1);
        let exec = builder.build();
        let ppo = TSOAxioms::compute_ppo(&exec);
        // W→R to same address IS in ppo
        assert!(ppo.get(0, 1));
    }

    #[test]
    fn test_x86_model_builder() {
        let model = X86ModelChecker::build_memory_model();
        assert_eq!(model.name, "x86-TSO");
        assert!(!model.derived_relations.is_empty());
    }

    #[test]
    fn test_litmus_sb() {
        let test = X86LitmusBuilder::store_buffering();
        assert_eq!(test.thread_count(), 2);
        assert_eq!(test.name, "SB");
    }

    #[test]
    fn test_litmus_mp() {
        let test = X86LitmusBuilder::message_passing();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_litmus_to_tso_programs() {
        let test = X86LitmusBuilder::store_buffering();
        let programs = X86LitmusBuilder::to_tso_programs(&test);
        assert_eq!(programs.len(), 2);
    }

    #[test]
    fn test_tso_instruction_display() {
        let t = TSOTransition::Read { thread: 0, addr: 0x100, reg: 0, value: 42 };
        let s = format!("{}", t);
        assert!(s.contains("T0"));
    }

    #[test]
    fn test_tso_execution_trace() {
        let state = TSOState::new(2);
        let trace = TSOExecutionTrace::new(state);
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
    }

    #[test]
    fn test_tso_final_state_display() {
        let fs = TSOFinalState {
            memory: BTreeMap::from([(0x100, 1), (0x200, 2)]),
            registers: vec![BTreeMap::from([(0, 1)]), BTreeMap::from([(0, 0)])],
        };
        let s = format!("{}", fs);
        assert!(!s.is_empty());
    }
}
