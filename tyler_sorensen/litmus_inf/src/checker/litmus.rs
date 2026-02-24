//! Litmus test representation and standard test library.
//!
//! Provides data structures for defining litmus tests (threads with
//! memory instructions and expected outcomes) plus a library of standard
//! tests (SB, MP, LB, IRIW, ISA2, WRC, 2+2W) including GPU-scoped variants.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

use super::execution::{
    Address, Value, EventId, ExecutionGraph, ExecutionGraphBuilder,
    OpType, Scope as ExecScope,
};

// ---------------------------------------------------------------------------
// Ordering / Scope
// ---------------------------------------------------------------------------

/// Memory ordering for instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
    /// GPU CTA-scoped acquire.
    AcquireCTA,
    /// GPU CTA-scoped release.
    ReleaseCTA,
    /// GPU GPU-scoped acquire.
    AcquireGPU,
    /// GPU GPU-scoped release.
    ReleaseGPU,
    /// GPU System-scoped acquire.
    AcquireSystem,
    /// GPU System-scoped release.
    ReleaseSystem,
}

impl Ordering {
    /// Map to execution-level scope.
    pub fn scope(&self) -> ExecScope {
        match self {
            Self::AcquireCTA | Self::ReleaseCTA => ExecScope::CTA,
            Self::AcquireGPU | Self::ReleaseGPU => ExecScope::GPU,
            Self::AcquireSystem | Self::ReleaseSystem => ExecScope::System,
            Self::SeqCst => ExecScope::System,
            _ => ExecScope::None,
        }
    }

    pub fn is_acquire(&self) -> bool {
        matches!(self,
            Self::Acquire | Self::AcqRel | Self::SeqCst |
            Self::AcquireCTA | Self::AcquireGPU | Self::AcquireSystem
        )
    }

    pub fn is_release(&self) -> bool {
        matches!(self,
            Self::Release | Self::AcqRel | Self::SeqCst |
            Self::ReleaseCTA | Self::ReleaseGPU | Self::ReleaseSystem
        )
    }
}

impl fmt::Display for Ordering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relaxed => write!(f, "rlx"),
            Self::Acquire => write!(f, "acq"),
            Self::Release => write!(f, "rel"),
            Self::AcqRel  => write!(f, "acq_rel"),
            Self::SeqCst  => write!(f, "sc"),
            Self::AcquireCTA => write!(f, "acq.cta"),
            Self::ReleaseCTA => write!(f, "rel.cta"),
            Self::AcquireGPU => write!(f, "acq.gpu"),
            Self::ReleaseGPU => write!(f, "rel.gpu"),
            Self::AcquireSystem => write!(f, "acq.sys"),
            Self::ReleaseSystem => write!(f, "rel.sys"),
        }
    }
}

/// GPU scope for fence instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    CTA,
    GPU,
    System,
    None,
}

impl Scope {
    pub fn to_exec_scope(&self) -> ExecScope {
        match self {
            Self::CTA => ExecScope::CTA,
            Self::GPU => ExecScope::GPU,
            Self::System => ExecScope::System,
            Self::None => ExecScope::None,
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CTA    => write!(f, ".cta"),
            Self::GPU    => write!(f, ".gpu"),
            Self::System => write!(f, ".sys"),
            Self::None   => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

/// Thread-local register identifier.
pub type RegId = usize;

// ---------------------------------------------------------------------------
// Instruction
// ---------------------------------------------------------------------------

/// A single instruction in a litmus test thread.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Instruction {
    /// Load value from memory into register.
    Load {
        reg: RegId,
        addr: Address,
        ordering: Ordering,
    },
    /// Store value to memory.
    Store {
        addr: Address,
        value: Value,
        ordering: Ordering,
    },
    /// Memory fence.
    Fence {
        ordering: Ordering,
        scope: Scope,
    },
    /// Read-modify-write (atomically read old value into reg, write new value).
    RMW {
        reg: RegId,
        addr: Address,
        value: Value,
        ordering: Ordering,
    },
    /// Unconditional branch (label index).
    Branch {
        label: usize,
    },
    /// Label definition.
    Label {
        id: usize,
    },
    /// Conditional branch (branch if register == expected).
    BranchCond {
        reg: RegId,
        expected: Value,
        label: usize,
    },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { reg, addr, ordering } =>
                write!(f, "r{} := load.{}({:#x})", reg, ordering, addr),
            Self::Store { addr, value, ordering } =>
                write!(f, "store.{}({:#x}, {})", ordering, addr, value),
            Self::Fence { ordering, scope } =>
                write!(f, "fence.{}{}", ordering, scope),
            Self::RMW { reg, addr, value, ordering } =>
                write!(f, "r{} := rmw.{}({:#x}, {})", reg, ordering, addr, value),
            Self::Branch { label } =>
                write!(f, "br L{}", label),
            Self::Label { id } =>
                write!(f, "L{}:", id),
            Self::BranchCond { reg, expected, label } =>
                write!(f, "br r{} == {} -> L{}", reg, expected, label),
        }
    }
}

// ---------------------------------------------------------------------------
// Thread
// ---------------------------------------------------------------------------

/// A thread in a litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    pub id: usize,
    pub instructions: Vec<Instruction>,
}

impl Thread {
    pub fn new(id: usize) -> Self {
        Self { id, instructions: Vec::new() }
    }

    pub fn with_instructions(id: usize, instrs: Vec<Instruction>) -> Self {
        Self { id, instructions: instrs }
    }

    pub fn add(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    pub fn load(&mut self, reg: RegId, addr: Address, ordering: Ordering) {
        self.add(Instruction::Load { reg, addr, ordering });
    }

    pub fn store(&mut self, addr: Address, value: Value, ordering: Ordering) {
        self.add(Instruction::Store { addr, value, ordering });
    }

    pub fn fence(&mut self, ordering: Ordering, scope: Scope) {
        self.add(Instruction::Fence { ordering, scope });
    }

    pub fn rmw(&mut self, reg: RegId, addr: Address, value: Value, ordering: Ordering) {
        self.add(Instruction::RMW { reg, addr, value, ordering });
    }

    /// Count memory operations (loads, stores, RMWs — not fences/branches).
    pub fn memory_op_count(&self) -> usize {
        self.instructions.iter().filter(|i| matches!(i,
            Instruction::Load { .. } | Instruction::Store { .. } | Instruction::RMW { .. }
        )).count()
    }

    /// Get all addresses accessed.
    pub fn accessed_addresses(&self) -> Vec<Address> {
        let mut addrs: Vec<Address> = self.instructions.iter().filter_map(|i| match i {
            Instruction::Load { addr, .. }
            | Instruction::Store { addr, .. }
            | Instruction::RMW { addr, .. } => Some(*addr),
            _ => None,
        }).collect();
        addrs.sort();
        addrs.dedup();
        addrs
    }
}

impl fmt::Display for Thread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "T{}:", self.id)?;
        for instr in &self.instructions {
            writeln!(f, "  {}", instr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// An expected final-state outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// Final register values: (thread, register) → value.
    pub registers: HashMap<(usize, RegId), Value>,
    /// Final memory values: address → value.
    pub memory: HashMap<Address, Value>,
}

impl Hash for Outcome {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut reg_entries: Vec<_> = self.registers.iter().collect();
        reg_entries.sort();
        for (k, v) in reg_entries {
            k.hash(state);
            v.hash(state);
        }
        let mut mem_entries: Vec<_> = self.memory.iter().collect();
        mem_entries.sort();
        for (k, v) in mem_entries {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl Outcome {
    pub fn new() -> Self {
        Self { registers: HashMap::new(), memory: HashMap::new() }
    }

    pub fn with_reg(mut self, thread: usize, reg: RegId, val: Value) -> Self {
        self.registers.insert((thread, reg), val);
        self
    }

    pub fn with_mem(mut self, addr: Address, val: Value) -> Self {
        self.memory.insert(addr, val);
        self
    }

    /// Check whether this outcome is matched by a given execution state.
    pub fn matches(&self, reg_state: &HashMap<(usize, RegId), Value>,
                   mem_state: &HashMap<Address, Value>) -> bool {
        for (&(tid, reg), &expected) in &self.registers {
            match reg_state.get(&(tid, reg)) {
                Some(&v) if v == expected => {}
                _ => return false,
            }
        }
        for (&addr, &expected) in &self.memory {
            match mem_state.get(&addr) {
                Some(&v) if v == expected => {}
                _ => return false,
            }
        }
        true
    }
}

impl Default for Outcome {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        let mut reg_keys: Vec<_> = self.registers.keys().collect();
        reg_keys.sort();
        for &(tid, reg) in &reg_keys {
            parts.push(format!("T{}:r{}={}", tid, reg, self.registers[&(*tid, *reg)]));
        }
        let mut mem_keys: Vec<_> = self.memory.keys().collect();
        mem_keys.sort();
        for &addr in &mem_keys {
            parts.push(format!("[{:#x}]={}", addr, self.memory[&addr]));
        }
        write!(f, "{}", parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// LitmusOutcome — observed or expected
// ---------------------------------------------------------------------------

/// Whether an outcome is expected to be observable or forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LitmusOutcome {
    /// The outcome is allowed by the model.
    Allowed,
    /// The outcome is forbidden by the model (a bug if observed).
    Forbidden,
    /// The outcome is required (must be the only possibility).
    Required,
}

impl fmt::Display for LitmusOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allowed   => write!(f, "allowed"),
            Self::Forbidden => write!(f, "forbidden"),
            Self::Required  => write!(f, "required"),
        }
    }
}

// ---------------------------------------------------------------------------
// LitmusTest
// ---------------------------------------------------------------------------

/// A complete litmus test specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LitmusTest {
    pub name: String,
    pub threads: Vec<Thread>,
    pub initial_state: HashMap<Address, Value>,
    pub expected_outcomes: Vec<(Outcome, LitmusOutcome)>,
}

impl LitmusTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            threads: Vec::new(),
            initial_state: HashMap::new(),
            expected_outcomes: Vec::new(),
        }
    }

    pub fn add_thread(&mut self, thread: Thread) {
        self.threads.push(thread);
    }

    pub fn set_initial(&mut self, addr: Address, val: Value) {
        self.initial_state.insert(addr, val);
    }

    pub fn expect(&mut self, outcome: Outcome, kind: LitmusOutcome) {
        self.expected_outcomes.push((outcome, kind));
    }

    /// Number of threads.
    pub fn thread_count(&self) -> usize { self.threads.len() }

    /// Alias for thread_count.
    pub fn num_threads(&self) -> usize { self.threads.len() }

    /// Total number of events across all threads.
    pub fn num_events(&self) -> usize { self.total_instructions() }

    /// Total number of instructions across all threads.
    pub fn total_instructions(&self) -> usize {
        self.threads.iter().map(|t| t.instructions.len()).sum()
    }

    /// Total number of memory operations (loads + stores + RMWs).
    pub fn total_memory_ops(&self) -> usize {
        self.threads.iter().map(|t| t.memory_op_count()).sum()
    }

    /// All addresses accessed.
    pub fn all_addresses(&self) -> Vec<Address> {
        let mut addrs: Vec<Address> = self.threads.iter()
            .flat_map(|t| t.accessed_addresses())
            .collect();
        for &addr in self.initial_state.keys() {
            addrs.push(addr);
        }
        addrs.sort();
        addrs.dedup();
        addrs
    }

    /// Generate all possible execution graphs (enumerate rf × co choices).
    pub fn enumerate_executions(&self) -> Vec<(ExecutionGraph, HashMap<(usize, RegId), Value>, HashMap<Address, Value>)> {
        let events = self.generate_events();
        let base_graph = ExecutionGraph::new(events);
        let n = base_graph.len();

        if n == 0 {
            return vec![(base_graph, HashMap::new(), self.initial_state.clone())];
        }

        let addresses = self.all_addresses();
        let mut reads_by_addr: HashMap<Address, Vec<EventId>> = HashMap::new();
        let mut writes_by_addr: HashMap<Address, Vec<EventId>> = HashMap::new();

        for e in &base_graph.events {
            match e.op_type {
                OpType::Read => { reads_by_addr.entry(e.address).or_default().push(e.id); }
                OpType::Write => { writes_by_addr.entry(e.address).or_default().push(e.id); }
                OpType::RMW => {
                    reads_by_addr.entry(e.address).or_default().push(e.id);
                    writes_by_addr.entry(e.address).or_default().push(e.id);
                }
                OpType::Fence => {}
            }
        }

        // Build rf choice combos per address.
        let mut rf_per_addr_combos: Vec<Vec<Vec<(EventId, EventId, Value)>>> = Vec::new();
        let mut co_choices: Vec<Vec<Vec<(EventId, EventId)>>> = Vec::new();

        for &addr in &addresses {
            let reads = reads_by_addr.get(&addr).cloned().unwrap_or_default();
            let writes = writes_by_addr.get(&addr).cloned().unwrap_or_default();
            let init_val = self.initial_state.get(&addr).copied().unwrap_or(0);

            let mut per_read: Vec<Vec<(EventId, EventId, Value)>> = Vec::new();
            for &r in &reads {
                let mut choices = Vec::new();
                choices.push((EventId::MAX, r, init_val));
                for &w in &writes {
                    if w != r {
                        choices.push((w, r, base_graph.events[w].value));
                    }
                }
                per_read.push(choices);
            }
            rf_per_addr_combos.push(cartesian_product(&per_read));

            // CO: enumerate all total orders on writes.
            let write_perms = permutations(&writes);
            let mut co_for_addr = Vec::new();
            for perm in write_perms {
                let mut edges = Vec::new();
                for i in 0..perm.len() {
                    for j in i + 1..perm.len() {
                        edges.push((perm[i], perm[j]));
                    }
                }
                co_for_addr.push(edges);
            }
            if co_for_addr.is_empty() {
                co_for_addr.push(Vec::new());
            }
            co_choices.push(co_for_addr);
        }

        let mut results = Vec::new();
        let rf_combos = cartesian_product(&rf_per_addr_combos);
        let co_combos = cartesian_product(&co_choices);

        for rf_combo in &rf_combos {
            for co_combo in &co_combos {
                let mut graph = base_graph.clone();
                let mut reg_state: HashMap<(usize, RegId), Value> = HashMap::new();

                for rf_per_addr in rf_combo {
                    for &(w, r, val) in rf_per_addr {
                        if w != EventId::MAX {
                            graph.rf.set(w, r, true);
                        }
                        let event = &graph.events[r];
                        if let Some(reg) = self.find_register_for_event(event.thread, event.po_index) {
                            reg_state.insert((event.thread, reg), val);
                        }
                    }
                }

                for co_per_addr in co_combo {
                    for &(w1, w2) in co_per_addr {
                        graph.co.set(w1, w2, true);
                    }
                }

                graph.derive_fr();

                // Add fr edges for reads-from-initial: if a read r reads from
                // the initial state, it has fr edges to ALL writes to that address
                // (since the initial state is implicitly co-before all writes).
                for rf_per_addr in rf_combo {
                    for &(w, r, _val) in rf_per_addr {
                        if w == EventId::MAX {
                            // r reads from initial → fr to all writes at same address.
                            let addr = graph.events[r].address;
                            if let Some(ws) = writes_by_addr.get(&addr) {
                                for &wr in ws {
                                    if wr != r {
                                        graph.fr.set(r, wr, true);
                                    }
                                }
                            }
                        }
                    }
                }

                let mut mem_state = self.initial_state.clone();
                for &addr in &addresses {
                    let writes = writes_by_addr.get(&addr).cloned().unwrap_or_default();
                    if writes.is_empty() { continue; }
                    let mut last_write = writes[0];
                    for &w in &writes[1..] {
                        if graph.co.get(last_write, w) {
                            last_write = w;
                        }
                    }
                    mem_state.insert(addr, graph.events[last_write].value);
                }

                results.push((graph, reg_state, mem_state));
            }
        }

        if results.is_empty() {
            results.push((base_graph, HashMap::new(), self.initial_state.clone()));
        }

        results
    }

    /// Generate events from the test instructions.
    fn generate_events(&self) -> Vec<super::execution::Event> {
        let mut events = Vec::new();
        let mut id = 0;

        for thread in &self.threads {
            let mut po_idx = 0;
            for instr in &thread.instructions {
                match instr {
                    Instruction::Load { addr, ordering, .. } => {
                        let mut e = super::execution::Event::new(
                            id, thread.id, OpType::Read, *addr, 0,
                        );
                        e.po_index = po_idx;
                        e.scope = ordering.scope();
                        events.push(e);
                        id += 1;
                        po_idx += 1;
                    }
                    Instruction::Store { addr, value, ordering } => {
                        let mut e = super::execution::Event::new(
                            id, thread.id, OpType::Write, *addr, *value,
                        );
                        e.po_index = po_idx;
                        e.scope = ordering.scope();
                        events.push(e);
                        id += 1;
                        po_idx += 1;
                    }
                    Instruction::Fence { ordering, scope } => {
                        let mut e = super::execution::Event::new(
                            id, thread.id, OpType::Fence, 0, 0,
                        );
                        e.po_index = po_idx;
                        e.scope = scope.to_exec_scope();
                        let _ = ordering;
                        events.push(e);
                        id += 1;
                        po_idx += 1;
                    }
                    Instruction::RMW { addr, value, ordering, .. } => {
                        let mut e = super::execution::Event::new(
                            id, thread.id, OpType::RMW, *addr, *value,
                        );
                        e.po_index = po_idx;
                        e.scope = ordering.scope();
                        events.push(e);
                        id += 1;
                        po_idx += 1;
                    }
                    Instruction::Branch { .. }
                    | Instruction::Label { .. }
                    | Instruction::BranchCond { .. } => {}
                }
            }
        }

        events
    }

    /// Find the register id for a load/RMW at a given thread and po_index.
    fn find_register_for_event(&self, thread_id: usize, po_index: usize) -> Option<RegId> {
        if thread_id >= self.threads.len() { return None; }
        let thread = &self.threads[thread_id];
        let mut mem_op_idx = 0;
        for instr in &thread.instructions {
            match instr {
                Instruction::Load { reg, .. } => {
                    if mem_op_idx == po_index { return Some(*reg); }
                    mem_op_idx += 1;
                }
                Instruction::Store { .. } => {
                    mem_op_idx += 1;
                }
                Instruction::RMW { reg, .. } => {
                    if mem_op_idx == po_index { return Some(*reg); }
                    mem_op_idx += 1;
                }
                Instruction::Fence { .. } => {
                    mem_op_idx += 1;
                }
                _ => {}
            }
        }
        None
    }
}

impl fmt::Display for LitmusTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Litmus test: {}", self.name)?;
        if !self.initial_state.is_empty() {
            write!(f, "Init: ")?;
            let mut entries: Vec<_> = self.initial_state.iter().collect();
            entries.sort_by_key(|(&addr, _)| addr);
            for (addr, val) in entries {
                write!(f, "[{:#x}]={} ", addr, val)?;
            }
            writeln!(f)?;
        }
        for t in &self.threads {
            write!(f, "{}", t)?;
        }
        for (outcome, kind) in &self.expected_outcomes {
            writeln!(f, "{}: {}", kind, outcome)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Standard litmus test library
// ---------------------------------------------------------------------------

/// Named address constants for readability.
const X: Address = 0x100;
const Y: Address = 0x200;
const Z: Address = 0x300;

/// Build the Store Buffer (SB / Dekker) litmus test.
///
/// ```text
/// T0:          T1:
///   W(x)=1       W(y)=1
///   R(y)=?       R(x)=?
/// ```
/// Forbidden under SC: r0==0 ∧ r1==0.
pub fn sb_test() -> LitmusTest {
    let mut test = LitmusTest::new("SB");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.load(0, Y, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    t1.load(0, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the Message Passing (MP) litmus test.
///
/// ```text
/// T0:          T1:
///   W(x)=1       R(y)=1
///   W(y)=1       R(x)=?
/// ```
/// Forbidden under SC: T1:r0==1 ∧ T1:r1==0.
pub fn mp_test() -> LitmusTest {
    let mut test = LitmusTest::new("MP");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the Load Buffering (LB) litmus test.
///
/// ```text
/// T0:          T1:
///   R(x)=?       R(y)=?
///   W(y)=1       W(x)=1
/// ```
/// Forbidden under SC: r0==1 ∧ r1==1 (causal cycle).
pub fn lb_test() -> LitmusTest {
    let mut test = LitmusTest::new("LB");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.store(X, 1, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the IRIW (Independent Reads of Independent Writes) litmus test.
///
/// ```text
/// T0: W(x)=1   T1: W(y)=1   T2: R(x)=1; R(y)=0   T3: R(y)=1; R(x)=0
/// ```
/// Forbidden under SC.
pub fn iriw_test() -> LitmusTest {
    let mut test = LitmusTest::new("IRIW");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, X, Ordering::Relaxed);
    t2.load(1, Y, Ordering::Relaxed);
    test.add_thread(t2);

    let mut t3 = Thread::new(3);
    t3.load(0, Y, Ordering::Relaxed);
    t3.load(1, X, Ordering::Relaxed);
    test.add_thread(t3);

    test.expect(
        Outcome::new()
            .with_reg(2, 0, 1).with_reg(2, 1, 0)
            .with_reg(3, 0, 1).with_reg(3, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the ISA2 litmus test.
pub fn isa2_test() -> LitmusTest {
    let mut test = LitmusTest::new("ISA2");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);
    test.set_initial(Z, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.store(Z, 1, Ordering::Relaxed);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, Z, Ordering::Relaxed);
    t2.load(1, X, Ordering::Relaxed);
    test.add_thread(t2);

    test.expect(
        Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(2, 0, 1)
            .with_reg(2, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the WRC (Write-Read Causality) litmus test.
pub fn wrc_test() -> LitmusTest {
    let mut test = LitmusTest::new("WRC");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, X, Ordering::Relaxed);
    t1.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, Y, Ordering::Relaxed);
    t2.load(1, X, Ordering::Relaxed);
    test.add_thread(t2);

    test.expect(
        Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(2, 0, 1)
            .with_reg(2, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build the 2+2W litmus test.
pub fn two_plus_two_w_test() -> LitmusTest {
    let mut test = LitmusTest::new("2+2W");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 2, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    t1.store(X, 2, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_mem(X, 1).with_mem(Y, 1),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build a GPU-scoped Message Passing test with CTA fence.
pub fn mp_cta_test() -> LitmusTest {
    let mut test = LitmusTest::new("MP+cta");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.fence(Ordering::Release, Scope::CTA);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.fence(Ordering::Acquire, Scope::CTA);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build a GPU-scoped Message Passing test with system fence.
pub fn mp_sys_test() -> LitmusTest {
    let mut test = LitmusTest::new("MP+sys");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.fence(Ordering::Release, Scope::System);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.fence(Ordering::Acquire, Scope::System);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build SB test with fences.
pub fn sb_fenced_test() -> LitmusTest {
    let mut test = LitmusTest::new("SB+fences");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.fence(Ordering::SeqCst, Scope::None);
    t0.load(0, Y, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    t1.fence(Ordering::SeqCst, Scope::None);
    t1.load(0, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Get all standard litmus tests.
pub fn standard_tests() -> Vec<LitmusTest> {
    vec![
        sb_test(),
        mp_test(),
        lb_test(),
        iriw_test(),
        isa2_test(),
        wrc_test(),
        two_plus_two_w_test(),
        mp_cta_test(),
        mp_sys_test(),
        sb_fenced_test(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cartesian product of a list of lists.
fn cartesian_product<T: Clone>(lists: &[Vec<T>]) -> Vec<Vec<T>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    let mut result = vec![vec![]];
    for list in lists {
        let mut new_result = Vec::new();
        for existing in &result {
            for item in list {
                let mut combo = existing.clone();
                combo.push(item.clone());
                new_result.push(combo);
            }
        }
        result = new_result;
    }
    result
}

/// Generate all permutations of a slice.
fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let mut rest: Vec<T> = items.to_vec();
        let item = rest.remove(i);
        for mut perm in permutations(&rest) {
            perm.insert(0, item.clone());
            result.push(perm);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordering_display() {
        assert_eq!(format!("{}", Ordering::Relaxed), "rlx");
        assert_eq!(format!("{}", Ordering::SeqCst), "sc");
        assert_eq!(format!("{}", Ordering::AcquireCTA), "acq.cta");
    }

    #[test]
    fn test_ordering_scope() {
        assert_eq!(Ordering::Relaxed.scope(), ExecScope::None);
        assert_eq!(Ordering::AcquireCTA.scope(), ExecScope::CTA);
        assert_eq!(Ordering::ReleaseGPU.scope(), ExecScope::GPU);
    }

    #[test]
    fn test_ordering_acquire_release() {
        assert!(Ordering::Acquire.is_acquire());
        assert!(!Ordering::Acquire.is_release());
        assert!(Ordering::Release.is_release());
        assert!(!Ordering::Release.is_acquire());
        assert!(Ordering::AcqRel.is_acquire());
        assert!(Ordering::AcqRel.is_release());
    }

    #[test]
    fn test_instruction_display() {
        let load = Instruction::Load { reg: 0, addr: 0x100, ordering: Ordering::Relaxed };
        assert!(format!("{}", load).contains("load"));

        let store = Instruction::Store { addr: 0x100, value: 42, ordering: Ordering::Release };
        assert!(format!("{}", store).contains("store"));

        let fence = Instruction::Fence { ordering: Ordering::SeqCst, scope: Scope::None };
        assert!(format!("{}", fence).contains("fence"));
    }

    #[test]
    fn test_thread_construction() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        t.load(0, 0x200, Ordering::Relaxed);
        t.fence(Ordering::SeqCst, Scope::None);
        assert_eq!(t.memory_op_count(), 2);
        assert_eq!(t.accessed_addresses(), vec![0x100, 0x200]);
    }

    #[test]
    fn test_thread_display() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        let s = format!("{}", t);
        assert!(s.contains("T0"));
    }

    #[test]
    fn test_outcome() {
        let o = Outcome::new().with_reg(0, 0, 1).with_mem(0x100, 42);
        let s = format!("{}", o);
        assert!(s.contains("r0=1"));
        assert!(s.contains("0x100"));
    }

    #[test]
    fn test_outcome_matches() {
        let o = Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 0);
        let mut regs = HashMap::new();
        regs.insert((0, 0), 1);
        regs.insert((1, 0), 0);
        let mem = HashMap::new();
        assert!(o.matches(&regs, &mem));

        regs.insert((1, 0), 1);
        assert!(!o.matches(&regs, &mem));
    }

    #[test]
    fn test_sb_test() {
        let test = sb_test();
        assert_eq!(test.name, "SB");
        assert_eq!(test.thread_count(), 2);
        assert_eq!(test.total_memory_ops(), 4);
        assert_eq!(test.all_addresses().len(), 2);
    }

    #[test]
    fn test_mp_test() {
        let test = mp_test();
        assert_eq!(test.name, "MP");
        assert_eq!(test.thread_count(), 2);
        assert_eq!(test.total_memory_ops(), 4);
    }

    #[test]
    fn test_lb_test() {
        let test = lb_test();
        assert_eq!(test.name, "LB");
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_iriw_test() {
        let test = iriw_test();
        assert_eq!(test.name, "IRIW");
        assert_eq!(test.thread_count(), 4);
    }

    #[test]
    fn test_isa2_test() {
        let test = isa2_test();
        assert_eq!(test.name, "ISA2");
        assert_eq!(test.thread_count(), 3);
    }

    #[test]
    fn test_wrc_test() {
        let test = wrc_test();
        assert_eq!(test.name, "WRC");
        assert_eq!(test.thread_count(), 3);
    }

    #[test]
    fn test_two_plus_two_w() {
        let test = two_plus_two_w_test();
        assert_eq!(test.name, "2+2W");
        assert_eq!(test.thread_count(), 2);
        assert_eq!(test.total_memory_ops(), 4);
    }

    #[test]
    fn test_gpu_tests() {
        let cta = mp_cta_test();
        assert_eq!(cta.name, "MP+cta");
        let sys = mp_sys_test();
        assert_eq!(sys.name, "MP+sys");
    }

    #[test]
    fn test_sb_fenced() {
        let test = sb_fenced_test();
        assert_eq!(test.name, "SB+fences");
        assert_eq!(test.total_instructions(), 6);
    }

    #[test]
    fn test_standard_tests() {
        let tests = standard_tests();
        assert!(tests.len() >= 10);
        for t in &tests {
            assert!(!t.name.is_empty());
            assert!(!t.threads.is_empty());
        }
    }

    #[test]
    fn test_litmus_display() {
        let test = sb_test();
        let s = format!("{}", test);
        assert!(s.contains("SB"));
        assert!(s.contains("T0"));
        assert!(s.contains("T1"));
    }

    #[test]
    fn test_litmus_outcome_display() {
        assert_eq!(format!("{}", LitmusOutcome::Allowed), "allowed");
        assert_eq!(format!("{}", LitmusOutcome::Forbidden), "forbidden");
    }

    #[test]
    fn test_cartesian_product() {
        let lists = vec![vec![1, 2], vec![3, 4]];
        let result = cartesian_product(&lists);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_permutations() {
        let items = vec![1, 2, 3];
        let perms = permutations(&items);
        assert_eq!(perms.len(), 6);
    }

    #[test]
    fn test_scope_display() {
        assert_eq!(format!("{}", Scope::CTA), ".cta");
        assert_eq!(format!("{}", Scope::None), "");
    }

    #[test]
    fn test_thread_rmw() {
        let mut t = Thread::new(0);
        t.rmw(0, 0x100, 42, Ordering::AcqRel);
        assert_eq!(t.memory_op_count(), 1);
    }

    #[test]
    fn test_instruction_branch() {
        let br = Instruction::Branch { label: 0 };
        let s = format!("{}", br);
        assert!(s.contains("L0"));

        let lbl = Instruction::Label { id: 1 };
        let s2 = format!("{}", lbl);
        assert!(s2.contains("L1"));

        let cond = Instruction::BranchCond { reg: 0, expected: 1, label: 2 };
        let s3 = format!("{}", cond);
        assert!(s3.contains("r0"));
    }

    #[test]
    fn test_generate_events() {
        let test = sb_test();
        let events = test.generate_events();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].op_type, OpType::Write);
        assert_eq!(events[1].op_type, OpType::Read);
        assert_eq!(events[2].op_type, OpType::Write);
        assert_eq!(events[3].op_type, OpType::Read);
    }

    #[test]
    fn test_enumerate_simple() {
        let mut test = LitmusTest::new("Simple");
        test.set_initial(X, 0);
        let mut t0 = Thread::new(0);
        t0.store(X, 1, Ordering::Relaxed);
        t0.load(0, X, Ordering::Relaxed);
        test.add_thread(t0);

        let execs = test.enumerate_executions();
        assert!(!execs.is_empty());
        for (g, _regs, _mem) in &execs {
            assert_eq!(g.len(), 2);
        }
    }
}
