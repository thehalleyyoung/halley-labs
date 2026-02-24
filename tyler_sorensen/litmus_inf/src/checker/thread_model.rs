//! Thread execution model for LITMUS∞.
//!
//! Implements thread state machines, interleaving enumeration,
//! thread-local state management, context switching semantics,
//! and thread scheduling strategies for concurrent program verification.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use super::execution::{EventId, ThreadId, Address, Value, OpType, Scope, ExecutionGraph, BitMatrix};
use super::litmus::{Instruction, Ordering as MemOrdering};

// ---------------------------------------------------------------------------
// Thread Status
// ---------------------------------------------------------------------------

/// Status of a thread in the execution model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreadStatus {
    /// Thread is ready to execute but not currently running.
    Ready,
    /// Thread is currently executing.
    Running,
    /// Thread is blocked waiting on a synchronization primitive.
    Blocked(BlockReason),
    /// Thread has completed all instructions.
    Completed,
    /// Thread has been aborted due to an error.
    Aborted,
}

impl Default for ThreadStatus {
    fn default() -> Self {
        ThreadStatus::Ready
    }
}

impl fmt::Display for ThreadStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreadStatus::Ready => write!(f, "Ready"),
            ThreadStatus::Running => write!(f, "Running"),
            ThreadStatus::Blocked(r) => write!(f, "Blocked({:?})", r),
            ThreadStatus::Completed => write!(f, "Completed"),
            ThreadStatus::Aborted => write!(f, "Aborted"),
        }
    }
}

/// Reason why a thread is blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockReason {
    /// Waiting on a mutex.
    Mutex(usize),
    /// Waiting on a barrier.
    Barrier(usize),
    /// Waiting on a condition variable.
    CondVar(usize),
    /// Waiting for a memory fence to complete.
    Fence,
    /// Waiting for a store buffer to drain.
    StoreBufferDrain,
}

// ---------------------------------------------------------------------------
// Register File
// ---------------------------------------------------------------------------

/// Thread-local register file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterFile {
    /// Register values indexed by register ID.
    registers: BTreeMap<usize, Value>,
    /// Number of general-purpose registers.
    num_registers: usize,
}

impl RegisterFile {
    /// Create a new register file with the given number of registers.
    pub fn new(num_registers: usize) -> Self {
        Self {
            registers: BTreeMap::new(),
            num_registers,
        }
    }

    /// Read a register value. Returns 0 if never written.
    pub fn read(&self, reg: usize) -> Value {
        self.registers.get(&reg).copied().unwrap_or(0)
    }

    /// Write a value to a register.
    pub fn write(&mut self, reg: usize, value: Value) {
        self.registers.insert(reg, value);
    }

    /// Check if a register has been written.
    pub fn is_defined(&self, reg: usize) -> bool {
        self.registers.contains_key(&reg)
    }

    /// Get all defined registers and their values.
    pub fn defined_registers(&self) -> &BTreeMap<usize, Value> {
        &self.registers
    }

    /// Clear all register values.
    pub fn clear(&mut self) {
        self.registers.clear();
    }

    /// Number of defined registers.
    pub fn defined_count(&self) -> usize {
        self.registers.len()
    }

    /// Snapshot the register file for comparison.
    pub fn snapshot(&self) -> Vec<(usize, Value)> {
        self.registers.iter().map(|(&r, &v)| (r, v)).collect()
    }
}

impl fmt::Display for RegisterFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, (reg, val)) in self.registers.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "r{}={}", reg, val)?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Store Buffer Entry
// ---------------------------------------------------------------------------

/// An entry in a thread's store buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreBufferEntry {
    /// Target memory address.
    pub address: Address,
    /// Value to be stored.
    pub value: Value,
    /// Memory ordering of the store.
    pub ordering: MemOrdering,
    /// Timestamp when the store was issued.
    pub timestamp: u64,
    /// Associated event ID in the execution graph.
    pub event_id: Option<EventId>,
}

impl StoreBufferEntry {
    pub fn new(address: Address, value: Value, ordering: MemOrdering) -> Self {
        Self {
            address,
            value,
            ordering,
            timestamp: 0,
            event_id: None,
        }
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn with_event(mut self, eid: EventId) -> Self {
        self.event_id = Some(eid);
        self
    }
}

// ---------------------------------------------------------------------------
// Store Buffer
// ---------------------------------------------------------------------------

/// Per-thread store buffer for relaxed memory model simulation.
#[derive(Debug, Clone)]
pub struct StoreBuffer {
    entries: VecDeque<StoreBufferEntry>,
    max_size: usize,
}

impl StoreBuffer {
    /// Create a new store buffer with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_size,
        }
    }

    /// Add a store to the buffer.
    pub fn push(&mut self, entry: StoreBufferEntry) -> bool {
        if self.entries.len() >= self.max_size {
            return false;
        }
        self.entries.push_back(entry);
        true
    }

    /// Try to forward a load from the store buffer.
    /// Returns the most recent store to the given address, if any.
    pub fn forward(&self, address: Address) -> Option<Value> {
        self.entries.iter().rev()
            .find(|e| e.address == address)
            .map(|e| e.value)
    }

    /// Drain the oldest entry from the store buffer.
    pub fn drain_oldest(&mut self) -> Option<StoreBufferEntry> {
        self.entries.pop_front()
    }

    /// Drain all entries for a specific address.
    pub fn drain_address(&mut self, address: Address) -> Vec<StoreBufferEntry> {
        let mut drained = Vec::new();
        self.entries.retain(|e| {
            if e.address == address {
                drained.push(e.clone());
                false
            } else {
                true
            }
        });
        drained
    }

    /// Drain all entries (flush the entire buffer).
    pub fn flush(&mut self) -> Vec<StoreBufferEntry> {
        self.entries.drain(..).collect()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_size
    }

    /// Number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get all entries as a slice.
    pub fn entries(&self) -> &VecDeque<StoreBufferEntry> {
        &self.entries
    }

    /// Check if there are pending stores to a given address.
    pub fn has_pending(&self, address: Address) -> bool {
        self.entries.iter().any(|e| e.address == address)
    }

    /// Count pending stores to a given address.
    pub fn pending_count(&self, address: Address) -> usize {
        self.entries.iter().filter(|e| e.address == address).count()
    }
}

// ---------------------------------------------------------------------------
// Thread Local Store
// ---------------------------------------------------------------------------

/// Thread-local state encompassing registers, local variables, and metadata.
#[derive(Debug, Clone)]
pub struct ThreadLocalStore {
    /// Thread identifier.
    pub thread_id: ThreadId,
    /// Register file.
    pub registers: RegisterFile,
    /// Local variables (thread-private memory).
    pub locals: HashMap<String, Value>,
    /// Store buffer for relaxed memory models.
    pub store_buffer: StoreBuffer,
    /// Memory ordering constraints accumulated.
    pub ordering_constraints: Vec<OrderingConstraint>,
    /// Thread-local event counter.
    pub event_counter: usize,
}

/// An ordering constraint between two events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderingConstraint {
    pub source: EventId,
    pub target: EventId,
    pub kind: OrderingKind,
}

/// Kind of ordering constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderingKind {
    ProgramOrder,
    Dependency,
    Fence,
    Acquire,
    Release,
    SeqCst,
}

impl ThreadLocalStore {
    /// Create a new thread-local store.
    pub fn new(thread_id: ThreadId, num_registers: usize, store_buffer_size: usize) -> Self {
        Self {
            thread_id,
            registers: RegisterFile::new(num_registers),
            locals: HashMap::new(),
            store_buffer: StoreBuffer::new(store_buffer_size),
            ordering_constraints: Vec::new(),
            event_counter: 0,
        }
    }

    /// Allocate the next event ID for this thread.
    pub fn next_event_id(&mut self) -> usize {
        let id = self.event_counter;
        self.event_counter += 1;
        id
    }

    /// Set a local variable.
    pub fn set_local(&mut self, name: &str, value: Value) {
        self.locals.insert(name.to_string(), value);
    }

    /// Get a local variable.
    pub fn get_local(&self, name: &str) -> Option<Value> {
        self.locals.get(name).copied()
    }

    /// Add an ordering constraint.
    pub fn add_constraint(&mut self, source: EventId, target: EventId, kind: OrderingKind) {
        self.ordering_constraints.push(OrderingConstraint {
            source, target, kind,
        });
    }

    /// Reset the thread-local store to initial state.
    pub fn reset(&mut self) {
        self.registers.clear();
        self.locals.clear();
        self.store_buffer.flush();
        self.ordering_constraints.clear();
        self.event_counter = 0;
    }

    /// Snapshot the current state.
    pub fn snapshot(&self) -> ThreadLocalSnapshot {
        ThreadLocalSnapshot {
            thread_id: self.thread_id,
            registers: self.registers.snapshot(),
            locals: self.locals.clone(),
            store_buffer_size: self.store_buffer.len(),
            event_counter: self.event_counter,
        }
    }
}

/// A snapshot of thread-local state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadLocalSnapshot {
    pub thread_id: ThreadId,
    pub registers: Vec<(usize, Value)>,
    pub locals: HashMap<String, Value>,
    pub store_buffer_size: usize,
    pub event_counter: usize,
}

// ---------------------------------------------------------------------------
// Thread State
// ---------------------------------------------------------------------------

/// Complete state of a thread during execution.
#[derive(Debug, Clone)]
pub struct ThreadState {
    /// Thread identifier.
    pub id: ThreadId,
    /// Current program counter (index into instruction list).
    pub pc: usize,
    /// Thread status.
    pub status: ThreadStatus,
    /// Instructions to execute.
    pub instructions: Vec<Instruction>,
    /// Thread-local store.
    pub local_store: ThreadLocalStore,
    /// Events generated by this thread.
    pub events: Vec<EventId>,
    /// Global timestamp for ordering.
    pub global_timestamp: u64,
    /// GPU scope for this thread.
    pub scope: Scope,
    /// CTA (Cooperative Thread Array) ID for GPU models.
    pub cta_id: usize,
    /// Warp ID within the CTA.
    pub warp_id: usize,
}

impl ThreadState {
    /// Create a new thread state.
    pub fn new(id: ThreadId, instructions: Vec<Instruction>) -> Self {
        Self {
            id,
            pc: 0,
            status: ThreadStatus::Ready,
            instructions,
            local_store: ThreadLocalStore::new(id, 32, 16),
            events: Vec::new(),
            global_timestamp: 0,
            scope: Scope::None,
            cta_id: 0,
            warp_id: 0,
        }
    }

    /// Create a GPU thread with scope information.
    pub fn new_gpu(id: ThreadId, instructions: Vec<Instruction>, scope: Scope, cta_id: usize, warp_id: usize) -> Self {
        Self {
            id,
            pc: 0,
            status: ThreadStatus::Ready,
            instructions,
            local_store: ThreadLocalStore::new(id, 32, 16),
            events: Vec::new(),
            global_timestamp: 0,
            scope,
            cta_id,
            warp_id,
        }
    }

    /// Check if the thread can execute more instructions.
    pub fn can_execute(&self) -> bool {
        self.status == ThreadStatus::Ready || self.status == ThreadStatus::Running
    }

    /// Check if the thread has completed.
    pub fn is_completed(&self) -> bool {
        self.status == ThreadStatus::Completed
    }

    /// Check if the thread is blocked.
    pub fn is_blocked(&self) -> bool {
        matches!(self.status, ThreadStatus::Blocked(_))
    }

    /// Get the current instruction, if any.
    pub fn current_instruction(&self) -> Option<&Instruction> {
        self.instructions.get(self.pc)
    }

    /// Advance the program counter.
    pub fn advance_pc(&mut self) {
        self.pc += 1;
        if self.pc >= self.instructions.len() {
            self.status = ThreadStatus::Completed;
        }
    }

    /// Set the thread status to running.
    pub fn start(&mut self) {
        if self.status == ThreadStatus::Ready {
            self.status = ThreadStatus::Running;
        }
    }

    /// Block the thread.
    pub fn block(&mut self, reason: BlockReason) {
        self.status = ThreadStatus::Blocked(reason);
    }

    /// Unblock the thread.
    pub fn unblock(&mut self) {
        if self.is_blocked() {
            self.status = ThreadStatus::Ready;
        }
    }

    /// Abort the thread.
    pub fn abort(&mut self) {
        self.status = ThreadStatus::Aborted;
    }

    /// Number of remaining instructions.
    pub fn remaining_instructions(&self) -> usize {
        if self.pc >= self.instructions.len() {
            0
        } else {
            self.instructions.len() - self.pc
        }
    }

    /// Total number of instructions.
    pub fn total_instructions(&self) -> usize {
        self.instructions.len()
    }

    /// Progress as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.instructions.is_empty() {
            1.0
        } else {
            self.pc as f64 / self.instructions.len() as f64
        }
    }

    /// Record an event generated by this thread.
    pub fn record_event(&mut self, event_id: EventId) {
        self.events.push(event_id);
    }

    /// Reset the thread to initial state.
    pub fn reset(&mut self) {
        self.pc = 0;
        self.status = ThreadStatus::Ready;
        self.local_store.reset();
        self.events.clear();
        self.global_timestamp = 0;
    }
}

impl fmt::Display for ThreadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Thread {} [{}] pc={}/{} events={}",
            self.id, self.status, self.pc,
            self.instructions.len(), self.events.len())
    }
}

// ---------------------------------------------------------------------------
// Thread State Machine Transitions
// ---------------------------------------------------------------------------

/// A transition in the thread state machine.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreadTransition {
    /// Start executing.
    Start,
    /// Execute an instruction and advance.
    Execute(usize),
    /// Block on a synchronization primitive.
    Block(BlockReason),
    /// Unblock after synchronization.
    Unblock,
    /// Complete execution.
    Complete,
    /// Abort execution.
    Abort,
    /// Context switch (yield to another thread).
    ContextSwitch,
}

impl fmt::Display for ThreadTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Execute(pc) => write!(f, "exec@{}", pc),
            Self::Block(r) => write!(f, "block({:?})", r),
            Self::Unblock => write!(f, "unblock"),
            Self::Complete => write!(f, "complete"),
            Self::Abort => write!(f, "abort"),
            Self::ContextSwitch => write!(f, "ctx_switch"),
        }
    }
}

/// State machine for thread execution.
#[derive(Debug, Clone)]
pub struct ThreadStateMachine {
    /// Valid transitions from each status.
    transitions: HashMap<ThreadStatus, Vec<ThreadTransition>>,
}

impl ThreadStateMachine {
    /// Create the default thread state machine.
    pub fn new() -> Self {
        let mut transitions = HashMap::new();

        transitions.insert(ThreadStatus::Ready, vec![
            ThreadTransition::Start,
            ThreadTransition::Abort,
        ]);

        transitions.insert(ThreadStatus::Running, vec![
            ThreadTransition::Execute(0),
            ThreadTransition::Block(BlockReason::Fence),
            ThreadTransition::Complete,
            ThreadTransition::ContextSwitch,
            ThreadTransition::Abort,
        ]);

        // Blocked threads can only unblock or abort.
        for reason in &[
            BlockReason::Mutex(0), BlockReason::Barrier(0),
            BlockReason::CondVar(0), BlockReason::Fence,
            BlockReason::StoreBufferDrain,
        ] {
            transitions.insert(ThreadStatus::Blocked(*reason), vec![
                ThreadTransition::Unblock,
                ThreadTransition::Abort,
            ]);
        }

        Self { transitions }
    }

    /// Check if a transition is valid from the given status.
    pub fn is_valid_transition(&self, from: ThreadStatus, transition: &ThreadTransition) -> bool {
        if let Some(valid) = self.transitions.get(&from) {
            valid.iter().any(|t| std::mem::discriminant(t) == std::mem::discriminant(transition))
        } else {
            false
        }
    }

    /// Apply a transition to a thread state. Returns true if successful.
    pub fn apply(&self, thread: &mut ThreadState, transition: &ThreadTransition) -> bool {
        match transition {
            ThreadTransition::Start => {
                if thread.status == ThreadStatus::Ready {
                    thread.status = ThreadStatus::Running;
                    true
                } else {
                    false
                }
            }
            ThreadTransition::Execute(_) => {
                if thread.status == ThreadStatus::Running && thread.pc < thread.instructions.len() {
                    thread.advance_pc();
                    true
                } else {
                    false
                }
            }
            ThreadTransition::Block(reason) => {
                if thread.status == ThreadStatus::Running {
                    thread.status = ThreadStatus::Blocked(*reason);
                    true
                } else {
                    false
                }
            }
            ThreadTransition::Unblock => {
                if thread.is_blocked() {
                    thread.status = ThreadStatus::Running;
                    true
                } else {
                    false
                }
            }
            ThreadTransition::Complete => {
                if thread.status == ThreadStatus::Running {
                    thread.status = ThreadStatus::Completed;
                    true
                } else {
                    false
                }
            }
            ThreadTransition::Abort => {
                thread.status = ThreadStatus::Aborted;
                true
            }
            ThreadTransition::ContextSwitch => {
                if thread.status == ThreadStatus::Running {
                    thread.status = ThreadStatus::Ready;
                    true
                } else {
                    false
                }
            }
        }
    }
}

impl Default for ThreadStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Context Switch
// ---------------------------------------------------------------------------

/// What happens to thread state during a context switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextSwitchPolicy {
    /// Preserve all state (sequential consistency).
    PreserveAll,
    /// Flush store buffer on switch (TSO-like).
    FlushStoreBuffer,
    /// Preserve store buffer (weak memory).
    PreserveStoreBuffer,
    /// Flush and barrier (strong ordering).
    FlushAndBarrier,
}

impl Default for ContextSwitchPolicy {
    fn default() -> Self {
        ContextSwitchPolicy::PreserveAll
    }
}

/// Context switch manager.
#[derive(Debug, Clone)]
pub struct ContextSwitchManager {
    policy: ContextSwitchPolicy,
    switch_count: usize,
    switch_history: Vec<(ThreadId, ThreadId, u64)>,
}

impl ContextSwitchManager {
    pub fn new(policy: ContextSwitchPolicy) -> Self {
        Self {
            policy,
            switch_count: 0,
            switch_history: Vec::new(),
        }
    }

    /// Perform a context switch from one thread to another.
    pub fn switch(&mut self, from: &mut ThreadState, to: &mut ThreadState, timestamp: u64) -> bool {
        if !from.can_execute() && !from.is_blocked() {
            return false;
        }

        match self.policy {
            ContextSwitchPolicy::PreserveAll => {
                // Nothing special needed.
            }
            ContextSwitchPolicy::FlushStoreBuffer => {
                from.local_store.store_buffer.flush();
            }
            ContextSwitchPolicy::PreserveStoreBuffer => {
                // Store buffer preserved across switch.
            }
            ContextSwitchPolicy::FlushAndBarrier => {
                from.local_store.store_buffer.flush();
                // Mark a barrier in ordering constraints.
            }
        }

        if from.status == ThreadStatus::Running {
            from.status = ThreadStatus::Ready;
        }
        to.status = ThreadStatus::Running;
        to.global_timestamp = timestamp;

        self.switch_count += 1;
        self.switch_history.push((from.id, to.id, timestamp));
        true
    }

    /// Get the number of context switches performed.
    pub fn switch_count(&self) -> usize {
        self.switch_count
    }

    /// Get the context switch history.
    pub fn history(&self) -> &[(ThreadId, ThreadId, u64)] {
        &self.switch_history
    }

    /// Reset the manager.
    pub fn reset(&mut self) {
        self.switch_count = 0;
        self.switch_history.clear();
    }
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

/// Scheduling strategy for thread interleaving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// Round-robin among ready threads.
    RoundRobin,
    /// Random selection among ready threads.
    Random,
    /// Priority-based (lower priority number = higher priority).
    Priority,
    /// Depth-first exploration of interleavings.
    DepthFirst,
    /// Breadth-first exploration of interleavings.
    BreadthFirst,
    /// Exhaustive enumeration of all interleavings.
    Exhaustive,
}

/// Thread priority assignment.
#[derive(Debug, Clone)]
pub struct ThreadPriority {
    priorities: HashMap<ThreadId, i32>,
    default_priority: i32,
}

impl ThreadPriority {
    pub fn new(default: i32) -> Self {
        Self {
            priorities: HashMap::new(),
            default_priority: default,
        }
    }

    pub fn set(&mut self, tid: ThreadId, priority: i32) {
        self.priorities.insert(tid, priority);
    }

    pub fn get(&self, tid: ThreadId) -> i32 {
        self.priorities.get(&tid).copied().unwrap_or(self.default_priority)
    }
}

impl Default for ThreadPriority {
    fn default() -> Self {
        Self::new(0)
    }
}

/// A scheduler that selects the next thread to execute.
#[derive(Debug, Clone)]
pub struct Scheduler {
    strategy: SchedulingStrategy,
    priorities: ThreadPriority,
    /// Current index for round-robin.
    rr_index: usize,
    /// RNG seed for random scheduling.
    rng_state: u64,
    /// History of scheduled threads.
    schedule_history: Vec<ThreadId>,
}

impl Scheduler {
    /// Create a new scheduler with the given strategy.
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            priorities: ThreadPriority::default(),
            rr_index: 0,
            rng_state: 12345,
            schedule_history: Vec::new(),
        }
    }

    /// Create a scheduler with a specific random seed.
    pub fn with_seed(strategy: SchedulingStrategy, seed: u64) -> Self {
        Self {
            strategy,
            priorities: ThreadPriority::default(),
            rr_index: 0,
            rng_state: seed,
            schedule_history: Vec::new(),
        }
    }

    /// Set thread priorities.
    pub fn set_priorities(&mut self, priorities: ThreadPriority) {
        self.priorities = priorities;
    }

    /// Select the next thread to run from the given list of ready threads.
    pub fn select_next(&mut self, ready_threads: &[ThreadId]) -> Option<ThreadId> {
        if ready_threads.is_empty() {
            return None;
        }

        let selected = match self.strategy {
            SchedulingStrategy::RoundRobin => {
                let idx = self.rr_index % ready_threads.len();
                self.rr_index += 1;
                ready_threads[idx]
            }
            SchedulingStrategy::Random => {
                // Simple xorshift64 PRNG.
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let idx = (self.rng_state as usize) % ready_threads.len();
                ready_threads[idx]
            }
            SchedulingStrategy::Priority => {
                *ready_threads.iter()
                    .min_by_key(|&&tid| self.priorities.get(tid))
                    .unwrap()
            }
            SchedulingStrategy::DepthFirst | SchedulingStrategy::BreadthFirst => {
                // For DFS/BFS, always pick the first ready thread.
                ready_threads[0]
            }
            SchedulingStrategy::Exhaustive => {
                // For exhaustive, the caller drives the enumeration.
                ready_threads[0]
            }
        };

        self.schedule_history.push(selected);
        Some(selected)
    }

    /// Get the schedule history.
    pub fn history(&self) -> &[ThreadId] {
        &self.schedule_history
    }

    /// Reset the scheduler state.
    pub fn reset(&mut self) {
        self.rr_index = 0;
        self.schedule_history.clear();
    }
}

// ---------------------------------------------------------------------------
// Interleaving Enumeration
// ---------------------------------------------------------------------------

/// A schedule: sequence of (thread_id, instruction_index) pairs.
pub type Schedule = Vec<(ThreadId, usize)>;

/// Enumerator for thread interleavings.
#[derive(Debug, Clone)]
pub struct InterleavingEnumerator {
    /// Number of threads.
    num_threads: usize,
    /// Number of instructions per thread.
    instructions_per_thread: Vec<usize>,
    /// Maximum number of interleavings to enumerate (0 = unlimited).
    max_interleavings: usize,
    /// Count of interleavings generated so far.
    count: usize,
    /// Whether to use partial order reduction.
    use_por: bool,
}

impl InterleavingEnumerator {
    /// Create a new enumerator for threads with given instruction counts.
    pub fn new(instructions_per_thread: Vec<usize>) -> Self {
        let num_threads = instructions_per_thread.len();
        Self {
            num_threads,
            instructions_per_thread,
            max_interleavings: 0,
            count: 0,
            use_por: false,
        }
    }

    /// Set a limit on the number of interleavings.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_interleavings = limit;
        self
    }

    /// Enable partial order reduction.
    pub fn with_por(mut self, enabled: bool) -> Self {
        self.use_por = enabled;
        self
    }

    /// Count the total number of possible interleavings (multinomial coefficient).
    pub fn total_interleavings(&self) -> u64 {
        let total: usize = self.instructions_per_thread.iter().sum();
        let mut result: u64 = 1;
        let mut remaining = total as u64;

        for &count in &self.instructions_per_thread {
            for i in 0..count as u64 {
                result = result.saturating_mul(remaining - i);
                result /= i + 1;
            }
            remaining -= count as u64;
        }
        result
    }

    /// Generate all interleavings (up to the limit).
    pub fn enumerate_all(&mut self) -> Vec<Schedule> {
        let mut results = Vec::new();
        let mut current = Vec::new();
        let mut progress: Vec<usize> = vec![0; self.num_threads];

        self.count = 0;
        self.enumerate_recursive(&mut progress, &mut current, &mut results);
        results
    }

    fn enumerate_recursive(
        &mut self,
        progress: &mut Vec<usize>,
        current: &mut Schedule,
        results: &mut Vec<Schedule>,
    ) {
        if self.max_interleavings > 0 && self.count >= self.max_interleavings {
            return;
        }

        // Check if all threads are done.
        let all_done = progress.iter().enumerate().all(|(i, &p)| p >= self.instructions_per_thread[i]);
        if all_done {
            results.push(current.clone());
            self.count += 1;
            return;
        }

        for tid in 0..self.num_threads {
            if progress[tid] < self.instructions_per_thread[tid] {
                let instr_idx = progress[tid];
                current.push((tid, instr_idx));
                progress[tid] += 1;

                self.enumerate_recursive(progress, current, results);

                progress[tid] -= 1;
                current.pop();

                if self.max_interleavings > 0 && self.count >= self.max_interleavings {
                    return;
                }
            }
        }
    }

    /// Generate interleavings lazily using an iterator.
    pub fn iter(&self) -> InterleavingIterator {
        InterleavingIterator::new(self.instructions_per_thread.clone(), self.max_interleavings)
    }

    /// Number of interleavings generated in the last enumeration.
    pub fn generated_count(&self) -> usize {
        self.count
    }
}

/// Iterator over thread interleavings.
pub struct InterleavingIterator {
    instructions_per_thread: Vec<usize>,
    max_interleavings: usize,
    count: usize,
    /// Stack for iterative DFS: (progress, schedule, next_thread_to_try).
    stack: Vec<(Vec<usize>, Schedule, usize)>,
    done: bool,
}

impl InterleavingIterator {
    fn new(instructions_per_thread: Vec<usize>, max: usize) -> Self {
        let num_threads = instructions_per_thread.len();
        let initial_progress = vec![0; num_threads];
        let mut iter = Self {
            instructions_per_thread,
            max_interleavings: max,
            count: 0,
            stack: Vec::new(),
            done: false,
        };
        iter.stack.push((initial_progress, Vec::new(), 0));
        iter
    }
}

impl Iterator for InterleavingIterator {
    type Item = Schedule;

    fn next(&mut self) -> Option<Schedule> {
        if self.done {
            return None;
        }
        if self.max_interleavings > 0 && self.count >= self.max_interleavings {
            return None;
        }

        while let Some((progress, schedule, next_tid)) = self.stack.pop() {
            let all_done = progress.iter().enumerate()
                .all(|(i, &p)| p >= self.instructions_per_thread[i]);

            if all_done {
                self.count += 1;
                return Some(schedule);
            }

            // Push remaining threads to try (in reverse order for DFS).
            let num_threads = self.instructions_per_thread.len();
            for tid in (next_tid..num_threads).rev() {
                if progress[tid] < self.instructions_per_thread[tid] {
                    let mut new_progress = progress.clone();
                    let mut new_schedule = schedule.clone();
                    let instr_idx = new_progress[tid];
                    new_schedule.push((tid, instr_idx));
                    new_progress[tid] += 1;
                    self.stack.push((new_progress, new_schedule, 0));
                }
            }
        }

        self.done = true;
        None
    }
}

// ---------------------------------------------------------------------------
// Bounded Interleaving
// ---------------------------------------------------------------------------

/// Bounded context-switching enumeration.
/// Only explores interleavings with at most `bound` context switches.
#[derive(Debug, Clone)]
pub struct BoundedInterleavingEnumerator {
    instructions_per_thread: Vec<usize>,
    /// Maximum number of context switches allowed.
    bound: usize,
    count: usize,
}

impl BoundedInterleavingEnumerator {
    pub fn new(instructions_per_thread: Vec<usize>, bound: usize) -> Self {
        Self {
            instructions_per_thread,
            bound,
            count: 0,
        }
    }

    /// Enumerate all interleavings with at most `bound` context switches.
    pub fn enumerate(&mut self) -> Vec<Schedule> {
        let mut results = Vec::new();
        let mut progress = vec![0usize; self.instructions_per_thread.len()];
        let mut schedule = Vec::new();
        self.count = 0;

        if !self.instructions_per_thread.is_empty() {
            for start_tid in 0..self.instructions_per_thread.len() {
                self.enumerate_bounded(
                    &mut progress, &mut schedule, &mut results,
                    Some(start_tid), self.bound,
                );
            }
        }
        results
    }

    fn enumerate_bounded(
        &mut self,
        progress: &mut Vec<usize>,
        schedule: &mut Schedule,
        results: &mut Vec<Schedule>,
        last_thread: Option<usize>,
        remaining_switches: usize,
    ) {
        let all_done = progress.iter().enumerate()
            .all(|(i, &p)| p >= self.instructions_per_thread[i]);
        if all_done {
            results.push(schedule.clone());
            self.count += 1;
            return;
        }

        for tid in 0..self.instructions_per_thread.len() {
            if progress[tid] >= self.instructions_per_thread[tid] {
                continue;
            }

            let is_switch = last_thread.map_or(false, |lt| lt != tid);
            if is_switch && remaining_switches == 0 {
                continue;
            }

            let new_remaining = if is_switch {
                remaining_switches - 1
            } else {
                remaining_switches
            };

            let instr_idx = progress[tid];
            schedule.push((tid, instr_idx));
            progress[tid] += 1;

            self.enumerate_bounded(progress, schedule, results, Some(tid), new_remaining);

            progress[tid] -= 1;
            schedule.pop();
        }
    }

    pub fn generated_count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Synchronization Primitives
// ---------------------------------------------------------------------------

/// A mutex for thread synchronization modeling.
#[derive(Debug, Clone)]
pub struct MutexModel {
    id: usize,
    owner: Option<ThreadId>,
    waiters: VecDeque<ThreadId>,
    lock_count: usize,
}

impl MutexModel {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            owner: None,
            waiters: VecDeque::new(),
            lock_count: 0,
        }
    }

    /// Try to acquire the mutex. Returns true if acquired.
    pub fn try_lock(&mut self, tid: ThreadId) -> bool {
        if self.owner.is_none() {
            self.owner = Some(tid);
            self.lock_count += 1;
            true
        } else if self.owner == Some(tid) {
            // Recursive lock.
            self.lock_count += 1;
            true
        } else {
            false
        }
    }

    /// Queue a thread as waiting for the mutex.
    pub fn wait(&mut self, tid: ThreadId) {
        if !self.waiters.contains(&tid) {
            self.waiters.push_back(tid);
        }
    }

    /// Release the mutex. Returns the next waiter to wake up, if any.
    pub fn unlock(&mut self, tid: ThreadId) -> Option<ThreadId> {
        if self.owner != Some(tid) {
            return None;
        }
        self.lock_count -= 1;
        if self.lock_count == 0 {
            self.owner = None;
            // Wake up the next waiter.
            if let Some(next) = self.waiters.pop_front() {
                self.owner = Some(next);
                self.lock_count = 1;
                Some(next)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn is_locked(&self) -> bool { self.owner.is_some() }
    pub fn owner(&self) -> Option<ThreadId> { self.owner }
    pub fn waiter_count(&self) -> usize { self.waiters.len() }
}

/// A barrier for thread synchronization modeling.
#[derive(Debug, Clone)]
pub struct BarrierModel {
    id: usize,
    required: usize,
    arrived: HashSet<ThreadId>,
    generation: usize,
}

impl BarrierModel {
    pub fn new(id: usize, required: usize) -> Self {
        Self {
            id,
            required,
            arrived: HashSet::new(),
            generation: 0,
        }
    }

    /// A thread arrives at the barrier. Returns true if the barrier is released.
    pub fn arrive(&mut self, tid: ThreadId) -> bool {
        self.arrived.insert(tid);
        if self.arrived.len() >= self.required {
            self.arrived.clear();
            self.generation += 1;
            true
        } else {
            false
        }
    }

    /// Check if a thread has arrived.
    pub fn has_arrived(&self, tid: ThreadId) -> bool {
        self.arrived.contains(&tid)
    }

    /// Number of threads that have arrived.
    pub fn arrived_count(&self) -> usize {
        self.arrived.len()
    }

    /// Number of threads still needed.
    pub fn remaining(&self) -> usize {
        self.required.saturating_sub(self.arrived.len())
    }

    /// Current generation (how many times the barrier has been released).
    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// A condition variable for thread synchronization modeling.
#[derive(Debug, Clone)]
pub struct CondVarModel {
    id: usize,
    waiters: VecDeque<ThreadId>,
    signal_count: usize,
}

impl CondVarModel {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            waiters: VecDeque::new(),
            signal_count: 0,
        }
    }

    /// Add a thread to the wait queue.
    pub fn wait(&mut self, tid: ThreadId) {
        if !self.waiters.contains(&tid) {
            self.waiters.push_back(tid);
        }
    }

    /// Signal one waiting thread. Returns the thread to wake up.
    pub fn signal(&mut self) -> Option<ThreadId> {
        self.signal_count += 1;
        self.waiters.pop_front()
    }

    /// Signal all waiting threads. Returns all threads to wake up.
    pub fn broadcast(&mut self) -> Vec<ThreadId> {
        self.signal_count += 1;
        self.waiters.drain(..).collect()
    }

    pub fn waiter_count(&self) -> usize { self.waiters.len() }
    pub fn signal_count(&self) -> usize { self.signal_count }
}

// ---------------------------------------------------------------------------
// Shared Memory State
// ---------------------------------------------------------------------------

/// Shared memory state visible to all threads.
#[derive(Debug, Clone)]
pub struct SharedMemoryState {
    /// Memory contents: address -> value.
    memory: HashMap<Address, Value>,
    /// Coherence order per address: address -> list of (writer_tid, value, timestamp).
    coherence: HashMap<Address, Vec<(ThreadId, Value, u64)>>,
    /// Global timestamp counter.
    timestamp: u64,
}

impl SharedMemoryState {
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            coherence: HashMap::new(),
            timestamp: 0,
        }
    }

    /// Read a value from memory.
    pub fn read(&self, address: Address) -> Value {
        self.memory.get(&address).copied().unwrap_or(0)
    }

    /// Write a value to memory.
    pub fn write(&mut self, address: Address, value: Value, writer: ThreadId) {
        self.timestamp += 1;
        self.memory.insert(address, value);
        self.coherence.entry(address).or_default()
            .push((writer, value, self.timestamp));
    }

    /// Get the coherence history for an address.
    pub fn coherence_history(&self, address: Address) -> &[(ThreadId, Value, u64)] {
        self.coherence.get(&address).map_or(&[], |v| v.as_slice())
    }

    /// Get all addresses that have been written.
    pub fn written_addresses(&self) -> Vec<Address> {
        self.memory.keys().copied().collect()
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Reset memory to initial state.
    pub fn reset(&mut self) {
        self.memory.clear();
        self.coherence.clear();
        self.timestamp = 0;
    }

    /// Initialize memory with given values.
    pub fn initialize(&mut self, values: &[(Address, Value)]) {
        for &(addr, val) in values {
            self.memory.insert(addr, val);
        }
    }
}

impl Default for SharedMemoryState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Thread Pool
// ---------------------------------------------------------------------------

/// A pool of threads for concurrent execution simulation.
#[derive(Debug, Clone)]
pub struct ThreadPool {
    /// All threads in the pool.
    threads: Vec<ThreadState>,
    /// Shared memory state.
    shared_memory: SharedMemoryState,
    /// Synchronization primitives.
    mutexes: HashMap<usize, MutexModel>,
    barriers: HashMap<usize, BarrierModel>,
    condvars: HashMap<usize, CondVarModel>,
    /// Scheduler.
    scheduler: Scheduler,
    /// Context switch manager.
    ctx_manager: ContextSwitchManager,
    /// Total steps executed.
    total_steps: usize,
    /// Maximum steps before timeout.
    max_steps: usize,
}

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            threads: Vec::new(),
            shared_memory: SharedMemoryState::new(),
            mutexes: HashMap::new(),
            barriers: HashMap::new(),
            condvars: HashMap::new(),
            scheduler: Scheduler::new(strategy),
            ctx_manager: ContextSwitchManager::new(ContextSwitchPolicy::PreserveAll),
            total_steps: 0,
            max_steps: 100_000,
        }
    }

    /// Add a thread to the pool.
    pub fn add_thread(&mut self, instructions: Vec<Instruction>) -> ThreadId {
        let tid = self.threads.len();
        self.threads.push(ThreadState::new(tid, instructions));
        tid
    }

    /// Add a GPU thread with scope information.
    pub fn add_gpu_thread(
        &mut self, instructions: Vec<Instruction>,
        scope: Scope, cta_id: usize, warp_id: usize,
    ) -> ThreadId {
        let tid = self.threads.len();
        self.threads.push(ThreadState::new_gpu(tid, instructions, scope, cta_id, warp_id));
        tid
    }

    /// Get a thread by ID.
    pub fn thread(&self, tid: ThreadId) -> Option<&ThreadState> {
        self.threads.get(tid)
    }

    /// Get a mutable thread by ID.
    pub fn thread_mut(&mut self, tid: ThreadId) -> Option<&mut ThreadState> {
        self.threads.get_mut(tid)
    }

    /// Get all thread IDs.
    pub fn thread_ids(&self) -> Vec<ThreadId> {
        (0..self.threads.len()).collect()
    }

    /// Get ready thread IDs.
    pub fn ready_threads(&self) -> Vec<ThreadId> {
        self.threads.iter()
            .filter(|t| t.can_execute())
            .map(|t| t.id)
            .collect()
    }

    /// Check if all threads have completed.
    pub fn all_completed(&self) -> bool {
        self.threads.iter().all(|t| t.is_completed())
    }

    /// Check if any thread is blocked (potential deadlock).
    pub fn has_deadlock(&self) -> bool {
        let active = self.threads.iter().filter(|t| !t.is_completed()).count();
        let blocked = self.threads.iter().filter(|t| t.is_blocked()).count();
        active > 0 && blocked == active
    }

    /// Number of threads in the pool.
    pub fn thread_count(&self) -> usize {
        self.threads.len()
    }

    /// Access shared memory.
    pub fn shared_memory(&self) -> &SharedMemoryState {
        &self.shared_memory
    }

    /// Access shared memory mutably.
    pub fn shared_memory_mut(&mut self) -> &mut SharedMemoryState {
        &mut self.shared_memory
    }

    /// Set the context switch policy.
    pub fn set_context_switch_policy(&mut self, policy: ContextSwitchPolicy) {
        self.ctx_manager = ContextSwitchManager::new(policy);
    }

    /// Set maximum steps.
    pub fn set_max_steps(&mut self, max: usize) {
        self.max_steps = max;
    }

    /// Get total steps executed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Add a mutex.
    pub fn add_mutex(&mut self, id: usize) {
        self.mutexes.insert(id, MutexModel::new(id));
    }

    /// Add a barrier.
    pub fn add_barrier(&mut self, id: usize, count: usize) {
        self.barriers.insert(id, BarrierModel::new(id, count));
    }

    /// Add a condition variable.
    pub fn add_condvar(&mut self, id: usize) {
        self.condvars.insert(id, CondVarModel::new(id));
    }

    /// Reset all threads and shared state.
    pub fn reset(&mut self) {
        for thread in &mut self.threads {
            thread.reset();
        }
        self.shared_memory.reset();
        self.scheduler.reset();
        self.ctx_manager.reset();
        self.total_steps = 0;
    }

    /// Execute one step: pick a ready thread and execute one instruction.
    /// Returns the thread ID and instruction index that was executed.
    pub fn step(&mut self) -> Option<(ThreadId, usize)> {
        if self.total_steps >= self.max_steps {
            return None;
        }
        if self.all_completed() || self.has_deadlock() {
            return None;
        }

        let ready = self.ready_threads();
        if ready.is_empty() {
            return None;
        }

        let tid = self.scheduler.select_next(&ready)?;
        let thread = &mut self.threads[tid];
        if thread.status == ThreadStatus::Ready {
            thread.status = ThreadStatus::Running;
        }

        let pc = thread.pc;
        thread.advance_pc();

        if thread.pc >= thread.instructions.len() {
            thread.status = ThreadStatus::Completed;
        } else {
            thread.status = ThreadStatus::Ready;
        }

        self.total_steps += 1;
        Some((tid, pc))
    }

    /// Run to completion with the current scheduling strategy.
    /// Returns the schedule (sequence of thread/instruction pairs).
    pub fn run_to_completion(&mut self) -> Schedule {
        let mut schedule = Vec::new();
        while let Some(step) = self.step() {
            schedule.push(step);
        }
        schedule
    }

    /// Get the final register values for all threads.
    pub fn final_registers(&self) -> HashMap<ThreadId, Vec<(usize, Value)>> {
        self.threads.iter()
            .map(|t| (t.id, t.local_store.registers.snapshot()))
            .collect()
    }

    /// Get the final memory state.
    pub fn final_memory(&self) -> HashMap<Address, Value> {
        self.shared_memory.memory.clone()
    }
}

// ---------------------------------------------------------------------------
// Execution Trace
// ---------------------------------------------------------------------------

/// A recorded execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Sequence of steps: (thread_id, instruction_index, timestamp).
    pub steps: Vec<TraceStep>,
    /// Final register values per thread.
    pub final_registers: HashMap<ThreadId, Vec<(usize, Value)>>,
    /// Final memory state.
    pub final_memory: HashMap<Address, Value>,
    /// Context switch count.
    pub context_switches: usize,
    /// Total steps.
    pub total_steps: usize,
}

/// A single step in an execution trace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceStep {
    pub thread_id: ThreadId,
    pub instruction_index: usize,
    pub timestamp: u64,
    pub op_type: TraceOpType,
    pub address: Option<Address>,
    pub value: Option<Value>,
}

/// Type of operation in a trace step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceOpType {
    Load,
    Store,
    Fence,
    RMW,
    Branch,
    Nop,
}

impl ExecutionTrace {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_registers: HashMap::new(),
            final_memory: HashMap::new(),
            context_switches: 0,
            total_steps: 0,
        }
    }

    /// Add a step to the trace.
    pub fn add_step(&mut self, step: TraceStep) {
        if let Some(last) = self.steps.last() {
            if last.thread_id != step.thread_id {
                self.context_switches += 1;
            }
        }
        self.steps.push(step);
        self.total_steps += 1;
    }

    /// Get steps for a specific thread.
    pub fn thread_steps(&self, tid: ThreadId) -> Vec<&TraceStep> {
        self.steps.iter().filter(|s| s.thread_id == tid).collect()
    }

    /// Get the thread interleaving pattern.
    pub fn interleaving_pattern(&self) -> Vec<ThreadId> {
        self.steps.iter().map(|s| s.thread_id).collect()
    }

    /// Count steps per thread.
    pub fn steps_per_thread(&self) -> HashMap<ThreadId, usize> {
        let mut counts = HashMap::new();
        for step in &self.steps {
            *counts.entry(step.thread_id).or_insert(0) += 1;
        }
        counts
    }

    /// Detect data races in the trace (writes to same address from different threads
    /// without intervening synchronization).
    pub fn detect_races(&self) -> Vec<(TraceStep, TraceStep)> {
        let mut races = Vec::new();
        let mut last_access: HashMap<Address, (ThreadId, TraceOpType, usize)> = HashMap::new();

        for (i, step) in self.steps.iter().enumerate() {
            if let Some(addr) = step.address {
                if let Some(&(prev_tid, prev_op, prev_idx)) = last_access.get(&addr) {
                    if prev_tid != step.thread_id {
                        let is_race = matches!(
                            (prev_op, step.op_type),
                            (TraceOpType::Store, _) | (_, TraceOpType::Store)
                        );
                        if is_race {
                            races.push((self.steps[prev_idx].clone(), step.clone()));
                        }
                    }
                }
                last_access.insert(addr, (step.thread_id, step.op_type, i));
            }
        }
        races
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExecutionTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Execution Trace ({} steps, {} context switches)",
            self.total_steps, self.context_switches)?;
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(f, "  {:4}: T{} {:?} addr={:?} val={:?}",
                i, step.thread_id, step.op_type,
                step.address, step.value)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Thread Group (GPU-specific)
// ---------------------------------------------------------------------------

/// A group of threads sharing a scope (e.g., warp, CTA).
#[derive(Debug, Clone)]
pub struct ThreadGroup {
    /// Group identifier.
    pub id: usize,
    /// Scope of the group.
    pub scope: Scope,
    /// Thread IDs in this group.
    pub thread_ids: Vec<ThreadId>,
    /// Parent group (e.g., CTA is parent of warps).
    pub parent: Option<usize>,
    /// Child groups.
    pub children: Vec<usize>,
}

impl ThreadGroup {
    pub fn new(id: usize, scope: Scope) -> Self {
        Self {
            id,
            scope,
            thread_ids: Vec::new(),
            parent: None,
            children: Vec::new(),
        }
    }

    pub fn add_thread(&mut self, tid: ThreadId) {
        if !self.thread_ids.contains(&tid) {
            self.thread_ids.push(tid);
        }
    }

    pub fn contains(&self, tid: ThreadId) -> bool {
        self.thread_ids.contains(&tid)
    }

    pub fn size(&self) -> usize {
        self.thread_ids.len()
    }
}

/// Hierarchy of thread groups for GPU execution.
#[derive(Debug, Clone)]
pub struct ThreadGroupHierarchy {
    groups: Vec<ThreadGroup>,
    thread_to_group: HashMap<ThreadId, Vec<usize>>,
}

impl ThreadGroupHierarchy {
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            thread_to_group: HashMap::new(),
        }
    }

    /// Add a group and return its ID.
    pub fn add_group(&mut self, scope: Scope) -> usize {
        let id = self.groups.len();
        self.groups.push(ThreadGroup::new(id, scope));
        id
    }

    /// Add a thread to a group.
    pub fn assign_thread(&mut self, tid: ThreadId, group_id: usize) {
        if group_id < self.groups.len() {
            self.groups[group_id].add_thread(tid);
            self.thread_to_group.entry(tid).or_default().push(group_id);
        }
    }

    /// Set parent-child relationship.
    pub fn set_parent(&mut self, child_id: usize, parent_id: usize) {
        if child_id < self.groups.len() && parent_id < self.groups.len() {
            self.groups[child_id].parent = Some(parent_id);
            self.groups[parent_id].children.push(child_id);
        }
    }

    /// Check if two threads share a scope at the given level.
    pub fn share_scope(&self, t1: ThreadId, t2: ThreadId, scope: Scope) -> bool {
        let groups1 = self.thread_to_group.get(&t1);
        let groups2 = self.thread_to_group.get(&t2);

        if let (Some(g1), Some(g2)) = (groups1, groups2) {
            for &gid1 in g1 {
                if self.groups[gid1].scope == scope && g2.contains(&gid1) {
                    return true;
                }
            }
        }
        false
    }

    /// Get all groups a thread belongs to.
    pub fn groups_for_thread(&self, tid: ThreadId) -> Vec<&ThreadGroup> {
        self.thread_to_group.get(&tid)
            .map_or(Vec::new(), |gids| {
                gids.iter().filter_map(|&gid| self.groups.get(gid)).collect()
            })
    }

    /// Get a group by ID.
    pub fn group(&self, id: usize) -> Option<&ThreadGroup> {
        self.groups.get(id)
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}

impl Default for ThreadGroupHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Memory View (per-thread view of memory under weak models)
// ---------------------------------------------------------------------------

/// Per-thread view of memory, which may differ from the global state
/// under weak memory models.
#[derive(Debug, Clone)]
pub struct MemoryView {
    /// Thread's view of each address.
    view: HashMap<Address, Value>,
    /// Timestamp of last observation per address.
    timestamps: HashMap<Address, u64>,
}

impl MemoryView {
    pub fn new() -> Self {
        Self {
            view: HashMap::new(),
            timestamps: HashMap::new(),
        }
    }

    /// Read from this thread's view.
    pub fn read(&self, address: Address) -> Value {
        self.view.get(&address).copied().unwrap_or(0)
    }

    /// Update this thread's view.
    pub fn update(&mut self, address: Address, value: Value, timestamp: u64) {
        self.view.insert(address, value);
        self.timestamps.insert(address, timestamp);
    }

    /// Synchronize this view with global memory for specific addresses.
    pub fn sync_from_global(&mut self, global: &SharedMemoryState, addresses: &[Address]) {
        for &addr in addresses {
            let val = global.read(addr);
            let ts = global.timestamp();
            self.view.insert(addr, val);
            self.timestamps.insert(addr, ts);
        }
    }

    /// Synchronize all addresses from global memory.
    pub fn sync_all(&mut self, global: &SharedMemoryState) {
        let addrs: Vec<_> = global.written_addresses();
        self.sync_from_global(global, &addrs);
    }

    /// Get the timestamp of last observation of an address.
    pub fn last_seen(&self, address: Address) -> Option<u64> {
        self.timestamps.get(&address).copied()
    }

    /// Check if this view is coherent with global memory for an address.
    pub fn is_coherent(&self, address: Address, global: &SharedMemoryState) -> bool {
        self.read(address) == global.read(address)
    }

    /// Get all stale entries (where local view differs from global).
    pub fn stale_entries(&self, global: &SharedMemoryState) -> Vec<(Address, Value, Value)> {
        let mut stale = Vec::new();
        for (&addr, &local_val) in &self.view {
            let global_val = global.read(addr);
            if local_val != global_val {
                stale.push((addr, local_val, global_val));
            }
        }
        stale
    }
}

impl Default for MemoryView {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Execution Configuration
// ---------------------------------------------------------------------------

/// Configuration for thread execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Number of general-purpose registers per thread.
    pub num_registers: usize,
    /// Store buffer size.
    pub store_buffer_size: usize,
    /// Context switch policy.
    pub context_switch_policy: ContextSwitchPolicy,
    /// Scheduling strategy.
    pub scheduling_strategy: SchedulingStrategy,
    /// Maximum steps before timeout.
    pub max_steps: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to record execution traces.
    pub record_trace: bool,
    /// Whether to detect data races.
    pub detect_races: bool,
    /// Bound for bounded context-switching (0 = unbounded).
    pub context_switch_bound: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            num_registers: 32,
            store_buffer_size: 16,
            context_switch_policy: ContextSwitchPolicy::PreserveAll,
            scheduling_strategy: SchedulingStrategy::RoundRobin,
            max_steps: 100_000,
            seed: 42,
            record_trace: false,
            detect_races: false,
            context_switch_bound: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution Engine
// ---------------------------------------------------------------------------

/// Main execution engine coordinating threads.
#[derive(Debug, Clone)]
pub struct ExecutionEngine {
    pool: ThreadPool,
    config: ExecutionConfig,
    traces: Vec<ExecutionTrace>,
}

impl ExecutionEngine {
    /// Create a new execution engine with the given configuration.
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            pool: ThreadPool::new(config.scheduling_strategy),
            config,
            traces: Vec::new(),
        }
    }

    /// Add a thread with given instructions.
    pub fn add_thread(&mut self, instructions: Vec<Instruction>) -> ThreadId {
        self.pool.add_thread(instructions)
    }

    /// Initialize shared memory.
    pub fn initialize_memory(&mut self, values: &[(Address, Value)]) {
        self.pool.shared_memory_mut().initialize(values);
    }

    /// Run a single execution and return the trace.
    pub fn run_single(&mut self) -> ExecutionTrace {
        self.pool.reset();
        let schedule = self.pool.run_to_completion();

        let mut trace = ExecutionTrace::new();
        for (tid, instr_idx) in &schedule {
            trace.add_step(TraceStep {
                thread_id: *tid,
                instruction_index: *instr_idx,
                timestamp: trace.total_steps as u64,
                op_type: TraceOpType::Nop,
                address: None,
                value: None,
            });
        }
        trace.final_registers = self.pool.final_registers();
        trace.final_memory = self.pool.final_memory();

        if self.config.record_trace {
            self.traces.push(trace.clone());
        }
        trace
    }

    /// Get the thread pool.
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }

    /// Get the thread pool mutably.
    pub fn pool_mut(&mut self) -> &mut ThreadPool {
        &mut self.pool
    }

    /// Get recorded traces.
    pub fn traces(&self) -> &[ExecutionTrace] {
        &self.traces
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instructions(count: usize) -> Vec<Instruction> {
        (0..count).map(|i| {
            Instruction::Store {
                addr: (i as u64) * 4,
                value: i as u64 + 1,
                ordering: MemOrdering::Relaxed,
            }
        }).collect()
    }

    // -- RegisterFile tests --

    #[test]
    fn test_register_file_read_write() {
        let mut rf = RegisterFile::new(8);
        assert_eq!(rf.read(0), 0);
        rf.write(0, 42);
        assert_eq!(rf.read(0), 42);
        rf.write(3, 100);
        assert_eq!(rf.read(3), 100);
    }

    #[test]
    fn test_register_file_defined() {
        let mut rf = RegisterFile::new(8);
        assert!(!rf.is_defined(0));
        rf.write(0, 1);
        assert!(rf.is_defined(0));
        assert_eq!(rf.defined_count(), 1);
    }

    #[test]
    fn test_register_file_clear() {
        let mut rf = RegisterFile::new(8);
        rf.write(0, 1);
        rf.write(1, 2);
        rf.clear();
        assert_eq!(rf.defined_count(), 0);
        assert_eq!(rf.read(0), 0);
    }

    #[test]
    fn test_register_file_snapshot() {
        let mut rf = RegisterFile::new(8);
        rf.write(0, 10);
        rf.write(2, 20);
        let snap = rf.snapshot();
        assert_eq!(snap, vec![(0, 10), (2, 20)]);
    }

    // -- StoreBuffer tests --

    #[test]
    fn test_store_buffer_push_forward() {
        let mut sb = StoreBuffer::new(4);
        let entry = StoreBufferEntry::new(0x100, 42, MemOrdering::Relaxed);
        assert!(sb.push(entry));
        assert_eq!(sb.forward(0x100), Some(42));
        assert_eq!(sb.forward(0x200), None);
    }

    #[test]
    fn test_store_buffer_forward_latest() {
        let mut sb = StoreBuffer::new(4);
        sb.push(StoreBufferEntry::new(0x100, 10, MemOrdering::Relaxed));
        sb.push(StoreBufferEntry::new(0x100, 20, MemOrdering::Relaxed));
        assert_eq!(sb.forward(0x100), Some(20));
    }

    #[test]
    fn test_store_buffer_full() {
        let mut sb = StoreBuffer::new(2);
        assert!(sb.push(StoreBufferEntry::new(0x100, 1, MemOrdering::Relaxed)));
        assert!(sb.push(StoreBufferEntry::new(0x200, 2, MemOrdering::Relaxed)));
        assert!(!sb.push(StoreBufferEntry::new(0x300, 3, MemOrdering::Relaxed)));
        assert!(sb.is_full());
    }

    #[test]
    fn test_store_buffer_drain() {
        let mut sb = StoreBuffer::new(4);
        sb.push(StoreBufferEntry::new(0x100, 1, MemOrdering::Relaxed));
        sb.push(StoreBufferEntry::new(0x200, 2, MemOrdering::Relaxed));
        let oldest = sb.drain_oldest().unwrap();
        assert_eq!(oldest.address, 0x100);
        assert_eq!(sb.len(), 1);
    }

    #[test]
    fn test_store_buffer_flush() {
        let mut sb = StoreBuffer::new(4);
        sb.push(StoreBufferEntry::new(0x100, 1, MemOrdering::Relaxed));
        sb.push(StoreBufferEntry::new(0x200, 2, MemOrdering::Relaxed));
        let flushed = sb.flush();
        assert_eq!(flushed.len(), 2);
        assert!(sb.is_empty());
    }

    // -- ThreadLocalStore tests --

    #[test]
    fn test_thread_local_store() {
        let mut tls = ThreadLocalStore::new(0, 8, 4);
        tls.registers.write(0, 42);
        tls.set_local("x", 100);
        assert_eq!(tls.registers.read(0), 42);
        assert_eq!(tls.get_local("x"), Some(100));
        assert_eq!(tls.get_local("y"), None);
    }

    #[test]
    fn test_thread_local_store_event_counter() {
        let mut tls = ThreadLocalStore::new(0, 8, 4);
        assert_eq!(tls.next_event_id(), 0);
        assert_eq!(tls.next_event_id(), 1);
        assert_eq!(tls.next_event_id(), 2);
    }

    #[test]
    fn test_thread_local_store_reset() {
        let mut tls = ThreadLocalStore::new(0, 8, 4);
        tls.registers.write(0, 42);
        tls.set_local("x", 100);
        tls.next_event_id();
        tls.reset();
        assert_eq!(tls.registers.read(0), 0);
        assert_eq!(tls.get_local("x"), None);
        assert_eq!(tls.event_counter, 0);
    }

    // -- ThreadState tests --

    #[test]
    fn test_thread_state_creation() {
        let instrs = make_instructions(3);
        let ts = ThreadState::new(0, instrs);
        assert_eq!(ts.id, 0);
        assert_eq!(ts.pc, 0);
        assert_eq!(ts.status, ThreadStatus::Ready);
        assert_eq!(ts.total_instructions(), 3);
    }

    #[test]
    fn test_thread_state_advance() {
        let instrs = make_instructions(2);
        let mut ts = ThreadState::new(0, instrs);
        ts.start();
        assert_eq!(ts.status, ThreadStatus::Running);
        ts.advance_pc();
        assert_eq!(ts.pc, 1);
        ts.advance_pc();
        assert_eq!(ts.status, ThreadStatus::Completed);
    }

    #[test]
    fn test_thread_state_block_unblock() {
        let mut ts = ThreadState::new(0, make_instructions(2));
        ts.start();
        ts.block(BlockReason::Mutex(0));
        assert!(ts.is_blocked());
        assert!(!ts.can_execute());
        ts.unblock();
        assert!(!ts.is_blocked());
        assert_eq!(ts.status, ThreadStatus::Ready);
    }

    #[test]
    fn test_thread_state_progress() {
        let mut ts = ThreadState::new(0, make_instructions(4));
        assert_eq!(ts.progress(), 0.0);
        ts.advance_pc();
        assert_eq!(ts.progress(), 0.25);
        ts.advance_pc();
        assert_eq!(ts.progress(), 0.5);
    }

    #[test]
    fn test_thread_state_remaining() {
        let mut ts = ThreadState::new(0, make_instructions(5));
        assert_eq!(ts.remaining_instructions(), 5);
        ts.advance_pc();
        assert_eq!(ts.remaining_instructions(), 4);
    }

    #[test]
    fn test_thread_state_reset() {
        let mut ts = ThreadState::new(0, make_instructions(3));
        ts.start();
        ts.advance_pc();
        ts.record_event(10);
        ts.reset();
        assert_eq!(ts.pc, 0);
        assert_eq!(ts.status, ThreadStatus::Ready);
        assert!(ts.events.is_empty());
    }

    // -- ThreadStateMachine tests --

    #[test]
    fn test_state_machine_transitions() {
        let sm = ThreadStateMachine::new();
        let mut ts = ThreadState::new(0, make_instructions(2));

        assert!(sm.apply(&mut ts, &ThreadTransition::Start));
        assert_eq!(ts.status, ThreadStatus::Running);

        assert!(sm.apply(&mut ts, &ThreadTransition::Execute(0)));
        assert!(sm.apply(&mut ts, &ThreadTransition::ContextSwitch));
        assert_eq!(ts.status, ThreadStatus::Ready);
    }

    #[test]
    fn test_state_machine_invalid_transition() {
        let sm = ThreadStateMachine::new();
        let mut ts = ThreadState::new(0, make_instructions(2));
        // Can't execute when not running.
        assert!(!sm.apply(&mut ts, &ThreadTransition::Execute(0)));
    }

    // -- ContextSwitchManager tests --

    #[test]
    fn test_context_switch_preserve_all() {
        let mut mgr = ContextSwitchManager::new(ContextSwitchPolicy::PreserveAll);
        let mut t0 = ThreadState::new(0, make_instructions(2));
        let mut t1 = ThreadState::new(1, make_instructions(2));
        t0.start();
        assert!(mgr.switch(&mut t0, &mut t1, 1));
        assert_eq!(t0.status, ThreadStatus::Ready);
        assert_eq!(t1.status, ThreadStatus::Running);
        assert_eq!(mgr.switch_count(), 1);
    }

    #[test]
    fn test_context_switch_flush() {
        let mut mgr = ContextSwitchManager::new(ContextSwitchPolicy::FlushStoreBuffer);
        let mut t0 = ThreadState::new(0, make_instructions(2));
        t0.start();
        t0.local_store.store_buffer.push(
            StoreBufferEntry::new(0x100, 42, MemOrdering::Relaxed)
        );
        let mut t1 = ThreadState::new(1, make_instructions(2));
        mgr.switch(&mut t0, &mut t1, 1);
        assert!(t0.local_store.store_buffer.is_empty());
    }

    // -- Scheduler tests --

    #[test]
    fn test_scheduler_round_robin() {
        let mut sched = Scheduler::new(SchedulingStrategy::RoundRobin);
        let ready = vec![0, 1, 2];
        assert_eq!(sched.select_next(&ready), Some(0));
        assert_eq!(sched.select_next(&ready), Some(1));
        assert_eq!(sched.select_next(&ready), Some(2));
        assert_eq!(sched.select_next(&ready), Some(0));
    }

    #[test]
    fn test_scheduler_priority() {
        let mut sched = Scheduler::new(SchedulingStrategy::Priority);
        let mut prio = ThreadPriority::new(10);
        prio.set(0, 5);
        prio.set(1, 1);
        prio.set(2, 3);
        sched.set_priorities(prio);
        let ready = vec![0, 1, 2];
        assert_eq!(sched.select_next(&ready), Some(1));
    }

    #[test]
    fn test_scheduler_empty() {
        let mut sched = Scheduler::new(SchedulingStrategy::RoundRobin);
        assert_eq!(sched.select_next(&[]), None);
    }

    // -- InterleavingEnumerator tests --

    #[test]
    fn test_interleaving_count() {
        let enumerator = InterleavingEnumerator::new(vec![2, 2]);
        // C(4,2) = 6 interleavings.
        assert_eq!(enumerator.total_interleavings(), 6);
    }

    #[test]
    fn test_interleaving_enumerate() {
        let mut enumerator = InterleavingEnumerator::new(vec![1, 1]);
        let interleavings = enumerator.enumerate_all();
        assert_eq!(interleavings.len(), 2);
    }

    #[test]
    fn test_interleaving_three_threads() {
        let mut enumerator = InterleavingEnumerator::new(vec![1, 1, 1]);
        let interleavings = enumerator.enumerate_all();
        // 3! = 6
        assert_eq!(interleavings.len(), 6);
    }

    #[test]
    fn test_interleaving_with_limit() {
        let mut enumerator = InterleavingEnumerator::new(vec![2, 2]).with_limit(3);
        let interleavings = enumerator.enumerate_all();
        assert_eq!(interleavings.len(), 3);
    }

    #[test]
    fn test_interleaving_iterator() {
        let enumerator = InterleavingEnumerator::new(vec![1, 1]);
        let count = enumerator.iter().count();
        assert_eq!(count, 2);
    }

    // -- BoundedInterleavingEnumerator tests --

    #[test]
    fn test_bounded_interleaving_zero_switches() {
        let mut enumerator = BoundedInterleavingEnumerator::new(vec![2, 2], 0);
        let interleavings = enumerator.enumerate();
        // With 0 context switches: T0T0T1T1 and T1T1T0T0
        assert_eq!(interleavings.len(), 2);
    }

    #[test]
    fn test_bounded_interleaving_one_switch() {
        let mut enumerator = BoundedInterleavingEnumerator::new(vec![1, 1], 1);
        let interleavings = enumerator.enumerate();
        // Both orderings are possible with ≤1 switch.
        assert_eq!(interleavings.len(), 2);
    }

    // -- MutexModel tests --

    #[test]
    fn test_mutex_lock_unlock() {
        let mut m = MutexModel::new(0);
        assert!(m.try_lock(0));
        assert!(m.is_locked());
        assert_eq!(m.owner(), Some(0));
        assert!(!m.try_lock(1));
        m.unlock(0);
        assert!(!m.is_locked());
    }

    #[test]
    fn test_mutex_waiter() {
        let mut m = MutexModel::new(0);
        m.try_lock(0);
        m.wait(1);
        m.wait(2);
        assert_eq!(m.waiter_count(), 2);
        let next = m.unlock(0);
        assert_eq!(next, Some(1));
        assert_eq!(m.owner(), Some(1));
    }

    // -- BarrierModel tests --

    #[test]
    fn test_barrier() {
        let mut b = BarrierModel::new(0, 3);
        assert!(!b.arrive(0));
        assert!(!b.arrive(1));
        assert!(b.arrive(2));
        assert_eq!(b.generation(), 1);
        assert_eq!(b.arrived_count(), 0);
    }

    // -- CondVarModel tests --

    #[test]
    fn test_condvar() {
        let mut cv = CondVarModel::new(0);
        cv.wait(0);
        cv.wait(1);
        assert_eq!(cv.waiter_count(), 2);
        assert_eq!(cv.signal(), Some(0));
        assert_eq!(cv.waiter_count(), 1);
        let woken = cv.broadcast();
        assert_eq!(woken, vec![1]);
        assert_eq!(cv.waiter_count(), 0);
    }

    // -- SharedMemoryState tests --

    #[test]
    fn test_shared_memory() {
        let mut mem = SharedMemoryState::new();
        assert_eq!(mem.read(0x100), 0);
        mem.write(0x100, 42, 0);
        assert_eq!(mem.read(0x100), 42);
        mem.write(0x100, 100, 1);
        assert_eq!(mem.read(0x100), 100);
    }

    #[test]
    fn test_shared_memory_coherence() {
        let mut mem = SharedMemoryState::new();
        mem.write(0x100, 1, 0);
        mem.write(0x100, 2, 1);
        let history = mem.coherence_history(0x100);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].1, 1);
        assert_eq!(history[1].1, 2);
    }

    // -- ThreadPool tests --

    #[test]
    fn test_thread_pool_basic() {
        let mut pool = ThreadPool::new(SchedulingStrategy::RoundRobin);
        let t0 = pool.add_thread(make_instructions(2));
        let t1 = pool.add_thread(make_instructions(2));
        assert_eq!(pool.thread_count(), 2);
        assert!(!pool.all_completed());
    }

    #[test]
    fn test_thread_pool_run() {
        let mut pool = ThreadPool::new(SchedulingStrategy::RoundRobin);
        pool.add_thread(make_instructions(1));
        pool.add_thread(make_instructions(1));
        let schedule = pool.run_to_completion();
        assert_eq!(schedule.len(), 2);
        assert!(pool.all_completed());
    }

    #[test]
    fn test_thread_pool_deadlock_detection() {
        let mut pool = ThreadPool::new(SchedulingStrategy::RoundRobin);
        let t0 = pool.add_thread(make_instructions(2));
        pool.threads[t0].status = ThreadStatus::Blocked(BlockReason::Mutex(0));
        assert!(pool.has_deadlock());
    }

    // -- MemoryView tests --

    #[test]
    fn test_memory_view() {
        let mut view = MemoryView::new();
        assert_eq!(view.read(0x100), 0);
        view.update(0x100, 42, 1);
        assert_eq!(view.read(0x100), 42);
        assert_eq!(view.last_seen(0x100), Some(1));
    }

    #[test]
    fn test_memory_view_coherence() {
        let mut view = MemoryView::new();
        let mut global = SharedMemoryState::new();
        global.write(0x100, 42, 0);
        view.update(0x100, 10, 0);
        assert!(!view.is_coherent(0x100, &global));
        view.sync_all(&global);
        assert!(view.is_coherent(0x100, &global));
    }

    // -- ThreadGroup tests --

    #[test]
    fn test_thread_group_hierarchy() {
        let mut hier = ThreadGroupHierarchy::new();
        let gpu = hier.add_group(Scope::GPU);
        let cta0 = hier.add_group(Scope::CTA);
        let cta1 = hier.add_group(Scope::CTA);
        hier.set_parent(cta0, gpu);
        hier.set_parent(cta1, gpu);

        hier.assign_thread(0, cta0);
        hier.assign_thread(1, cta0);
        hier.assign_thread(2, cta1);

        assert!(hier.share_scope(0, 1, Scope::CTA));
        assert!(!hier.share_scope(0, 2, Scope::CTA));
    }

    // -- ExecutionTrace tests --

    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new();
        trace.add_step(TraceStep {
            thread_id: 0, instruction_index: 0, timestamp: 0,
            op_type: TraceOpType::Store, address: Some(0x100), value: Some(1),
        });
        trace.add_step(TraceStep {
            thread_id: 1, instruction_index: 0, timestamp: 1,
            op_type: TraceOpType::Store, address: Some(0x100), value: Some(2),
        });
        assert_eq!(trace.context_switches, 1);
        assert_eq!(trace.total_steps, 2);
    }

    #[test]
    fn test_execution_trace_race_detection() {
        let mut trace = ExecutionTrace::new();
        trace.add_step(TraceStep {
            thread_id: 0, instruction_index: 0, timestamp: 0,
            op_type: TraceOpType::Store, address: Some(0x100), value: Some(1),
        });
        trace.add_step(TraceStep {
            thread_id: 1, instruction_index: 0, timestamp: 1,
            op_type: TraceOpType::Load, address: Some(0x100), value: Some(1),
        });
        let races = trace.detect_races();
        assert_eq!(races.len(), 1);
    }

    // -- ExecutionEngine tests --

    #[test]
    fn test_execution_engine() {
        let config = ExecutionConfig {
            record_trace: true,
            ..Default::default()
        };
        let mut engine = ExecutionEngine::new(config);
        engine.add_thread(make_instructions(2));
        engine.add_thread(make_instructions(2));
        let trace = engine.run_single();
        assert_eq!(trace.total_steps, 4);
    }
}
