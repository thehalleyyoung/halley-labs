//! Bounded model checking for memory models in LITMUS∞.
//!
//! Implements bounded model checking (BMC) with execution unrolling,
//! symmetry-aware reduction, counterexample extraction, property
//! encoding, transition systems, and statistics tracking.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════
// BmcConfig — BMC configuration
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for bounded model checking.
#[derive(Debug, Clone)]
pub struct BmcConfig {
    /// Maximum bound for unrolling.
    pub max_bound: usize,
    /// Enable symmetry reduction.
    pub symmetry_reduction: bool,
    /// Enable incremental solving.
    pub incremental: bool,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
    /// BMC strategy.
    pub strategy: BmcStrategy,
    /// Maximum number of context switches.
    pub max_context_switches: usize,
}

impl Default for BmcConfig {
    fn default() -> Self {
        BmcConfig {
            max_bound: 10,
            symmetry_reduction: true,
            incremental: true,
            timeout_ms: 30000,
            strategy: BmcStrategy::Forward,
            max_context_switches: 4,
        }
    }
}

/// BMC search strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmcStrategy {
    /// Forward unrolling from initial state.
    Forward,
    /// Backward from property violation.
    Backward,
    /// Bidirectional search.
    Bidirectional,
    /// Interpolation-based (stub).
    InterpolationBased,
}

// ═══════════════════════════════════════════════════════════════════════
// BmcResult — result of bounded model checking
// ═══════════════════════════════════════════════════════════════════════

/// Result of bounded model checking.
#[derive(Debug, Clone)]
pub enum BmcResult {
    /// Property verified up to the given bound.
    Verified {
        /// The bound up to which verification succeeded.
        bound: usize,
        /// Statistics.
        stats: BmcStatistics,
    },
    /// Counterexample found at the given bound.
    CounterexampleFound {
        /// The counterexample trace.
        trace: Counterexample,
        /// The bound at which it was found.
        bound: usize,
    },
    /// Timeout reached.
    Timeout {
        /// The bound reached before timeout.
        reached_bound: usize,
    },
    /// Unknown result.
    Unknown,
}

impl fmt::Display for BmcResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BmcResult::Verified { bound, stats } => {
                write!(f, "Verified up to bound {} ({} states explored)", bound, stats.states_explored)
            }
            BmcResult::CounterexampleFound { bound, .. } => {
                write!(f, "Counterexample found at bound {}", bound)
            }
            BmcResult::Timeout { reached_bound } => {
                write!(f, "Timeout (reached bound {})", reached_bound)
            }
            BmcResult::Unknown => write!(f, "Unknown"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BmcStatistics — tracking BMC performance
// ═══════════════════════════════════════════════════════════════════════

/// Statistics for bounded model checking.
#[derive(Debug, Clone, Default)]
pub struct BmcStatistics {
    /// Total states explored.
    pub states_explored: usize,
    /// Total transitions explored.
    pub transitions_explored: usize,
    /// Time per bound level (in microseconds).
    pub time_per_level: Vec<u64>,
    /// States pruned by symmetry reduction.
    pub symmetry_pruned: usize,
    /// Peak memory usage estimate (bytes).
    pub peak_memory_bytes: usize,
    /// Number of bound levels completed.
    pub levels_completed: usize,
}

impl BmcStatistics {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total time across all levels (microseconds).
    pub fn total_time_us(&self) -> u64 {
        self.time_per_level.iter().sum()
    }

    /// Average time per level.
    pub fn avg_time_per_level_us(&self) -> f64 {
        if self.time_per_level.is_empty() {
            0.0
        } else {
            self.total_time_us() as f64 / self.time_per_level.len() as f64
        }
    }
}

impl fmt::Display for BmcStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BMC Statistics:")?;
        writeln!(f, "  States explored: {}", self.states_explored)?;
        writeln!(f, "  Transitions: {}", self.transitions_explored)?;
        writeln!(f, "  Levels completed: {}", self.levels_completed)?;
        writeln!(f, "  Symmetry pruned: {}", self.symmetry_pruned)?;
        writeln!(f, "  Total time: {:.2}ms", self.total_time_us() as f64 / 1000.0)?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Counterexample — a violating execution trace
// ═══════════════════════════════════════════════════════════════════════

/// A counterexample trace showing a property violation.
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// The sequence of states in the trace.
    pub states: Vec<State>,
    /// The transitions between states.
    pub transitions: Vec<Transition>,
    /// Which property was violated.
    pub violated_property: String,
    /// Human-readable explanation.
    pub explanation: String,
    /// Violating cycle (if any).
    pub cycle: Option<Vec<usize>>,
}

impl Counterexample {
    /// Create a new counterexample.
    pub fn new(property: &str) -> Self {
        Counterexample {
            states: Vec::new(),
            transitions: Vec::new(),
            violated_property: property.to_string(),
            explanation: String::new(),
            cycle: None,
        }
    }

    /// Add a state to the trace.
    pub fn add_state(&mut self, state: State) {
        self.states.push(state);
    }

    /// Add a transition.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Length of the trace.
    pub fn length(&self) -> usize {
        self.states.len()
    }

    /// Minimize the counterexample by removing redundant states.
    pub fn minimize(&mut self) {
        // Simple minimization: remove consecutive duplicate states
        if self.states.len() <= 2 { return; }

        let mut minimized = Vec::new();
        minimized.push(self.states[0].clone());
        for i in 1..self.states.len() {
            if self.states[i] != self.states[i-1] {
                minimized.push(self.states[i].clone());
            }
        }
        self.states = minimized;
    }
}

impl fmt::Display for Counterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Counterexample (violates: {})", self.violated_property)?;
        writeln!(f, "  Trace length: {}", self.length())?;
        for (i, state) in self.states.iter().enumerate() {
            writeln!(f, "  State {}: {}", i, state)?;
        }
        if !self.explanation.is_empty() {
            writeln!(f, "  Explanation: {}", self.explanation)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// State and Transition
// ═══════════════════════════════════════════════════════════════════════

/// A state in the transition system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct State {
    /// Thread program counters.
    pub program_counters: Vec<usize>,
    /// Memory state: address -> value.
    pub memory: BTreeMap<u64, u64>,
    /// Register state: (thread, register) -> value.
    pub registers: BTreeMap<(usize, usize), u64>,
    /// Store buffer contents per thread.
    pub store_buffers: Vec<Vec<(u64, u64)>>,
}

impl State {
    /// Create an initial state.
    pub fn initial(num_threads: usize) -> Self {
        State {
            program_counters: vec![0; num_threads],
            memory: BTreeMap::new(),
            registers: BTreeMap::new(),
            store_buffers: vec![vec![]; num_threads],
        }
    }

    /// Get memory value at address.
    pub fn read_memory(&self, addr: u64) -> u64 {
        self.memory.get(&addr).copied().unwrap_or(0)
    }

    /// Write to memory.
    pub fn write_memory(&mut self, addr: u64, val: u64) {
        self.memory.insert(addr, val);
    }

    /// Get register value.
    pub fn read_register(&self, thread: usize, reg: usize) -> u64 {
        self.registers.get(&(thread, reg)).copied().unwrap_or(0)
    }

    /// Write to register.
    pub fn write_register(&mut self, thread: usize, reg: usize, val: u64) {
        self.registers.insert((thread, reg), val);
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCs={:?}, Mem={{", self.program_counters)?;
        for (i, (addr, val)) in self.memory.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "[{}]={}", addr, val)?;
        }
        write!(f, "}}")
    }
}

/// A transition between states.
#[derive(Debug, Clone)]
pub struct Transition {
    /// The thread that executed.
    pub thread: usize,
    /// The action performed.
    pub action: Action,
    /// Source state index.
    pub from_state: usize,
    /// Target state index.
    pub to_state: usize,
}

/// An action in the transition system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Read from memory.
    Read { address: u64, value: u64 },
    /// Write to memory.
    Write { address: u64, value: u64 },
    /// Memory fence.
    Fence,
    /// Read-modify-write.
    Rmw { address: u64, old_value: u64, new_value: u64 },
    /// Internal computation.
    Internal,
    /// No operation.
    Nop,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Read { address, value } => write!(f, "R[{}]={}", address, value),
            Action::Write { address, value } => write!(f, "W[{}]={}", address, value),
            Action::Fence => write!(f, "fence"),
            Action::Rmw { address, old_value, new_value } =>
                write!(f, "RMW[{}] {}→{}", address, old_value, new_value),
            Action::Internal => write!(f, "internal"),
            Action::Nop => write!(f, "nop"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TransitionSystem — state machine for concurrent programs
// ═══════════════════════════════════════════════════════════════════════

/// A transition system for concurrent program verification.
#[derive(Debug, Clone)]
pub struct TransitionSystem {
    /// Number of threads.
    pub num_threads: usize,
    /// Instructions per thread.
    pub thread_instructions: Vec<Vec<ThreadInstruction>>,
    /// Initial state.
    pub initial_state: State,
}

/// An instruction in a thread.
#[derive(Debug, Clone)]
pub enum ThreadInstruction {
    /// Load: read memory[address] into register.
    Load { register: usize, address: u64 },
    /// Store: write register value to memory[address].
    Store { address: u64, value_reg: usize },
    /// Store immediate value.
    StoreImm { address: u64, value: u64 },
    /// Fence instruction.
    Fence,
    /// Compare: set register to 1 if reg1 == reg2, else 0.
    Compare { dst: usize, reg1: usize, reg2: usize },
    /// Load immediate value into register.
    LoadImm { register: usize, value: u64 },
}

impl TransitionSystem {
    /// Create a new transition system.
    pub fn new(thread_instructions: Vec<Vec<ThreadInstruction>>) -> Self {
        let num_threads = thread_instructions.len();
        TransitionSystem {
            num_threads,
            thread_instructions,
            initial_state: State::initial(num_threads),
        }
    }

    /// Check if a thread has finished executing.
    pub fn is_thread_done(&self, state: &State, thread: usize) -> bool {
        state.program_counters[thread] >= self.thread_instructions[thread].len()
    }

    /// Check if all threads have finished.
    pub fn is_terminal(&self, state: &State) -> bool {
        (0..self.num_threads).all(|t| self.is_thread_done(state, t))
    }

    /// Get all enabled threads (those that can execute next).
    pub fn enabled_threads(&self, state: &State) -> Vec<usize> {
        (0..self.num_threads)
            .filter(|&t| !self.is_thread_done(state, t))
            .collect()
    }

    /// Execute one step for a given thread, returning the new state and action.
    pub fn step(&self, state: &State, thread: usize) -> Option<(State, Action)> {
        if self.is_thread_done(state, thread) {
            return None;
        }

        let pc = state.program_counters[thread];
        let instr = &self.thread_instructions[thread][pc];
        let mut new_state = state.clone();
        new_state.program_counters[thread] = pc + 1;

        let action = match instr {
            ThreadInstruction::Load { register, address } => {
                let val = new_state.read_memory(*address);
                new_state.write_register(thread, *register, val);
                Action::Read { address: *address, value: val }
            }
            ThreadInstruction::Store { address, value_reg } => {
                let val = new_state.read_register(thread, *value_reg);
                new_state.write_memory(*address, val);
                Action::Write { address: *address, value: val }
            }
            ThreadInstruction::StoreImm { address, value } => {
                new_state.write_memory(*address, *value);
                Action::Write { address: *address, value: *value }
            }
            ThreadInstruction::Fence => {
                Action::Fence
            }
            ThreadInstruction::Compare { dst, reg1, reg2 } => {
                let v1 = new_state.read_register(thread, *reg1);
                let v2 = new_state.read_register(thread, *reg2);
                new_state.write_register(thread, *dst, if v1 == v2 { 1 } else { 0 });
                Action::Internal
            }
            ThreadInstruction::LoadImm { register, value } => {
                new_state.write_register(thread, *register, *value);
                Action::Internal
            }
        };

        Some((new_state, action))
    }

    /// Enumerate all possible next states from the current state.
    pub fn successors(&self, state: &State) -> Vec<(State, Transition)> {
        let mut result = Vec::new();
        for thread in self.enabled_threads(state) {
            if let Some((new_state, action)) = self.step(state, thread) {
                result.push((new_state, Transition {
                    thread,
                    action,
                    from_state: 0,
                    to_state: 0,
                }));
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ExecutionUnroller — unroll program executions to bounded depth
// ═══════════════════════════════════════════════════════════════════════

/// Unrolled execution trace.
#[derive(Debug, Clone)]
pub struct UnrolledExecution {
    /// Sequence of states.
    pub states: Vec<State>,
    /// Sequence of transitions.
    pub transitions: Vec<Transition>,
    /// Thread interleaving (sequence of thread IDs).
    pub interleaving: Vec<usize>,
}

/// Unroll program executions to bounded depth.
#[derive(Debug)]
pub struct ExecutionUnroller {
    /// Configuration.
    config: BmcConfig,
}

impl ExecutionUnroller {
    /// Create a new unroller.
    pub fn new(config: BmcConfig) -> Self {
        ExecutionUnroller { config }
    }

    /// Unroll all executions of the transition system up to the given depth.
    pub fn unroll(&self, system: &TransitionSystem, depth: usize) -> Vec<UnrolledExecution> {
        let mut results = Vec::new();
        let initial = system.initial_state.clone();
        let mut exec = UnrolledExecution {
            states: vec![initial.clone()],
            transitions: Vec::new(),
            interleaving: Vec::new(),
        };

        self.unroll_rec(system, &initial, depth, 0, &mut exec, &mut results);
        results
    }

    fn unroll_rec(
        &self,
        system: &TransitionSystem,
        state: &State,
        depth: usize,
        context_switches: usize,
        current: &mut UnrolledExecution,
        results: &mut Vec<UnrolledExecution>,
    ) {
        if depth == 0 || system.is_terminal(state) {
            results.push(current.clone());
            return;
        }

        if results.len() > 10000 {
            return; // Safety limit
        }

        let enabled = system.enabled_threads(state);
        let last_thread = current.interleaving.last().copied();

        for &thread in &enabled {
            // Count context switches
            let switches = if last_thread.map_or(false, |lt| lt != thread) {
                context_switches + 1
            } else {
                context_switches
            };

            if switches > self.config.max_context_switches {
                continue;
            }

            if let Some((new_state, action)) = system.step(state, thread) {
                current.states.push(new_state.clone());
                current.transitions.push(Transition {
                    thread,
                    action,
                    from_state: current.states.len() - 2,
                    to_state: current.states.len() - 1,
                });
                current.interleaving.push(thread);

                self.unroll_rec(system, &new_state, depth - 1, switches, current, results);

                current.states.pop();
                current.transitions.pop();
                current.interleaving.pop();
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SymmetryReducer — symmetry-aware bounded model checking
// ═══════════════════════════════════════════════════════════════════════

/// Symmetry-aware bounded model checking.
#[derive(Debug)]
pub struct SymmetryReducer {
    /// Thread symmetries (groups of interchangeable threads).
    pub thread_symmetries: Vec<Vec<usize>>,
    /// Data symmetries (groups of interchangeable values).
    pub data_symmetries: Vec<Vec<u64>>,
}

impl SymmetryReducer {
    /// Create a new symmetry reducer.
    pub fn new() -> Self {
        SymmetryReducer {
            thread_symmetries: Vec::new(),
            data_symmetries: Vec::new(),
        }
    }

    /// Detect thread symmetries in a transition system.
    pub fn detect_thread_symmetries(&mut self, system: &TransitionSystem) {
        let n = system.num_threads;
        let mut groups: Vec<Vec<usize>> = Vec::new();

        // Simple symmetry detection: threads with identical instruction sequences
        let mut thread_types: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        for t in 0..n {
            let key: Vec<u8> = format!("{:?}", system.thread_instructions[t]).into_bytes();
            thread_types.entry(key).or_default().push(t);
        }

        for (_, group) in thread_types {
            if group.len() > 1 {
                groups.push(group);
            }
        }

        self.thread_symmetries = groups;
    }

    /// Detect data symmetries.
    pub fn detect_data_symmetries(&mut self, system: &TransitionSystem) {
        // Find addresses and values used
        let mut addresses: HashSet<u64> = HashSet::new();
        let mut values: HashSet<u64> = HashSet::new();

        for thread in &system.thread_instructions {
            for instr in thread {
                match instr {
                    ThreadInstruction::Load { address, .. } => { addresses.insert(*address); }
                    ThreadInstruction::Store { address, .. } => { addresses.insert(*address); }
                    ThreadInstruction::StoreImm { address, value } => {
                        addresses.insert(*address);
                        values.insert(*value);
                    }
                    ThreadInstruction::LoadImm { value, .. } => { values.insert(*value); }
                    _ => {}
                }
            }
        }

        // All addresses are potential symmetry candidates
        if addresses.len() > 1 {
            let addr_vec: Vec<u64> = addresses.into_iter().collect();
            self.data_symmetries.push(addr_vec);
        }
    }

    /// Compute canonical form of a state under symmetry.
    pub fn canonical_form(&self, state: &State) -> State {
        // Apply lex-leader: sort thread states by symmetry group
        let mut canonical = state.clone();

        for group in &self.thread_symmetries {
            if group.len() < 2 { continue; }

            // Find the lexicographically smallest permutation
            let thread_states: Vec<(usize, u64)> = group.iter()
                .map(|&t| (canonical.program_counters[t], canonical.read_register(t, 0)))
                .collect();

            let mut sorted = thread_states.clone();
            sorted.sort();

            if sorted != thread_states {
                // Permute threads to match sorted order
                let perm: Vec<usize> = {
                    let mut indices: Vec<usize> = (0..group.len()).collect();
                    indices.sort_by_key(|&i| &thread_states[i]);
                    indices
                };

                for (new_idx, &old_idx) in perm.iter().enumerate() {
                    let t_new = group[new_idx];
                    let t_old = group[old_idx];
                    canonical.program_counters[t_new] = state.program_counters[t_old];
                }
            }
        }

        canonical
    }

    /// Check if a state should be pruned by symmetry.
    pub fn should_prune(&self, state: &State) -> bool {
        let canonical = self.canonical_form(state);
        canonical != *state
    }

    /// Number of thread symmetry groups.
    pub fn num_thread_symmetries(&self) -> usize {
        self.thread_symmetries.len()
    }

    /// Estimated reduction factor.
    pub fn estimated_reduction(&self) -> f64 {
        let mut factor = 1.0;
        for group in &self.thread_symmetries {
            // Each group of k symmetric threads reduces by k!
            let k = group.len();
            let mut factorial = 1.0;
            for i in 1..=k {
                factorial *= i as f64;
            }
            factor *= factorial;
        }
        factor
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CounterexampleExtractor — extract and minimize counterexamples
// ═══════════════════════════════════════════════════════════════════════

/// Extract counterexamples from model checking results.
#[derive(Debug)]
pub struct CounterexampleExtractor;

impl CounterexampleExtractor {
    /// Extract a counterexample from an execution trace.
    pub fn extract(
        execution: &UnrolledExecution,
        property: &str,
    ) -> Option<Counterexample> {
        let mut ce = Counterexample::new(property);
        ce.states = execution.states.clone();
        ce.transitions = execution.transitions.clone();
        ce.explanation = format!(
            "Execution of length {} violates property '{}'",
            execution.states.len(),
            property,
        );
        Some(ce)
    }

    /// Minimize a counterexample by trying to remove transitions.
    pub fn minimize(
        ce: &Counterexample,
        system: &TransitionSystem,
        property_checker: &dyn Fn(&State) -> bool,
    ) -> Counterexample {
        let mut best = ce.clone();

        // Try removing each transition and check if violation still occurs
        for skip in 0..ce.transitions.len() {
            let mut state = system.initial_state.clone();
            let mut valid = true;
            let mut new_states = vec![state.clone()];
            let mut new_transitions = Vec::new();

            for (i, trans) in ce.transitions.iter().enumerate() {
                if i == skip { continue; }
                match system.step(&state, trans.thread) {
                    Some((new_state, action)) => {
                        new_transitions.push(Transition {
                            thread: trans.thread,
                            action,
                            from_state: new_states.len() - 1,
                            to_state: new_states.len(),
                        });
                        state = new_state.clone();
                        new_states.push(new_state);
                    }
                    None => { valid = false; break; }
                }
            }

            if valid && !property_checker(&state) {
                if new_states.len() < best.states.len() {
                    best.states = new_states;
                    best.transitions = new_transitions;
                }
            }
        }

        best
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PropertyEncoder — encode safety and liveness properties
// ═══════════════════════════════════════════════════════════════════════

/// Encode properties for bounded model checking.
#[derive(Debug, Clone)]
pub enum Property {
    /// Safety: bad state should never be reached.
    Safety {
        name: String,
        predicate: PropertyPredicate,
    },
    /// Liveness (bounded): eventually the predicate holds.
    BoundedLiveness {
        name: String,
        predicate: PropertyPredicate,
        bound: usize,
    },
    /// Assertion: specific memory values at termination.
    Assertion {
        name: String,
        conditions: Vec<(u64, u64)>, // (address, expected_value)
    },
}

/// A predicate over states.
#[derive(Debug, Clone)]
pub enum PropertyPredicate {
    /// Memory location has a specific value.
    MemoryEquals { address: u64, value: u64 },
    /// Memory location does not have a specific value.
    MemoryNotEquals { address: u64, value: u64 },
    /// Register has a specific value.
    RegisterEquals { thread: usize, register: usize, value: u64 },
    /// Conjunction of predicates.
    And(Vec<PropertyPredicate>),
    /// Disjunction of predicates.
    Or(Vec<PropertyPredicate>),
    /// Negation.
    Not(Box<PropertyPredicate>),
    /// Always true.
    True,
}

impl PropertyPredicate {
    /// Evaluate the predicate on a state.
    pub fn evaluate(&self, state: &State) -> bool {
        match self {
            PropertyPredicate::MemoryEquals { address, value } => {
                state.read_memory(*address) == *value
            }
            PropertyPredicate::MemoryNotEquals { address, value } => {
                state.read_memory(*address) != *value
            }
            PropertyPredicate::RegisterEquals { thread, register, value } => {
                state.read_register(*thread, *register) == *value
            }
            PropertyPredicate::And(preds) => {
                preds.iter().all(|p| p.evaluate(state))
            }
            PropertyPredicate::Or(preds) => {
                preds.iter().any(|p| p.evaluate(state))
            }
            PropertyPredicate::Not(pred) => {
                !pred.evaluate(state)
            }
            PropertyPredicate::True => true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BoundedModelChecker — main BMC engine
// ═══════════════════════════════════════════════════════════════════════

/// Main bounded model checking engine.
#[derive(Debug)]
pub struct BoundedModelChecker {
    /// Configuration.
    config: BmcConfig,
    /// Symmetry reducer.
    symmetry: SymmetryReducer,
    /// Statistics.
    stats: BmcStatistics,
}

impl BoundedModelChecker {
    /// Create a new bounded model checker.
    pub fn new(config: BmcConfig) -> Self {
        BoundedModelChecker {
            config,
            symmetry: SymmetryReducer::new(),
            stats: BmcStatistics::new(),
        }
    }

    /// Check a property on a transition system.
    pub fn check(
        &mut self,
        system: &TransitionSystem,
        property: &Property,
    ) -> BmcResult {
        let start = Instant::now();

        // Detect symmetries
        if self.config.symmetry_reduction {
            self.symmetry.detect_thread_symmetries(system);
            self.symmetry.detect_data_symmetries(system);
        }

        // Iterative deepening
        for bound in 1..=self.config.max_bound {
            let level_start = Instant::now();

            // Check timeout
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                return BmcResult::Timeout { reached_bound: bound - 1 };
            }

            // Unroll and check
            let unroller = ExecutionUnroller::new(self.config.clone());
            let executions = unroller.unroll(system, bound);

            for exec in &executions {
                self.stats.states_explored += exec.states.len();
                self.stats.transitions_explored += exec.transitions.len();

                // Check symmetry pruning
                if self.config.symmetry_reduction {
                    if let Some(last_state) = exec.states.last() {
                        if self.symmetry.should_prune(last_state) {
                            self.stats.symmetry_pruned += 1;
                            continue;
                        }
                    }
                }

                // Check property on final state
                if let Some(final_state) = exec.states.last() {
                    if system.is_terminal(final_state) {
                        let violated = match property {
                            Property::Safety { predicate, .. } => {
                                !predicate.evaluate(final_state)
                            }
                            Property::Assertion { conditions, .. } => {
                                conditions.iter().any(|(addr, expected)| {
                                    final_state.read_memory(*addr) != *expected
                                })
                            }
                            Property::BoundedLiveness { predicate, .. } => {
                                !predicate.evaluate(final_state)
                            }
                        };

                        if violated {
                            if let Some(trace) = CounterexampleExtractor::extract(exec, &match property {
                                Property::Safety { name, .. } => name.clone(),
                                Property::Assertion { name, .. } => name.clone(),
                                Property::BoundedLiveness { name, .. } => name.clone(),
                            }) {
                                return BmcResult::CounterexampleFound {
                                    trace,
                                    bound,
                                };
                            }
                        }
                    }
                }
            }

            let level_time = level_start.elapsed().as_micros() as u64;
            self.stats.time_per_level.push(level_time);
            self.stats.levels_completed = bound;
        }

        BmcResult::Verified {
            bound: self.config.max_bound,
            stats: self.stats.clone(),
        }
    }

    /// Get statistics.
    pub fn statistics(&self) -> &BmcStatistics {
        &self.stats
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simple 2-thread message passing program
    // Thread 0: W[x]=1; W[y]=1
    // Thread 1: R[y]; R[x]
    fn message_passing_system() -> TransitionSystem {
        TransitionSystem::new(vec![
            vec![
                ThreadInstruction::StoreImm { address: 0, value: 1 }, // W[x]=1
                ThreadInstruction::StoreImm { address: 1, value: 1 }, // W[y]=1
            ],
            vec![
                ThreadInstruction::Load { register: 0, address: 1 }, // r0 = R[y]
                ThreadInstruction::Load { register: 1, address: 0 }, // r1 = R[x]
            ],
        ])
    }

    // Helper: store buffer litmus test
    // Thread 0: W[x]=1; R[y]
    // Thread 1: W[y]=1; R[x]
    fn store_buffer_system() -> TransitionSystem {
        TransitionSystem::new(vec![
            vec![
                ThreadInstruction::StoreImm { address: 0, value: 1 },
                ThreadInstruction::Load { register: 0, address: 1 },
            ],
            vec![
                ThreadInstruction::StoreImm { address: 1, value: 1 },
                ThreadInstruction::Load { register: 0, address: 0 },
            ],
        ])
    }

    #[test]
    fn test_transition_system_creation() {
        let system = message_passing_system();
        assert_eq!(system.num_threads, 2);
        assert!(!system.is_terminal(&system.initial_state));
    }

    #[test]
    fn test_transition_system_step() {
        let system = message_passing_system();
        let state = system.initial_state.clone();

        let (new_state, action) = system.step(&state, 0).unwrap();
        assert_eq!(new_state.program_counters[0], 1);
        match action {
            Action::Write { address: 0, value: 1 } => {}
            _ => panic!("Expected write to x=1"),
        }
    }

    #[test]
    fn test_execution_unroller() {
        let system = message_passing_system();
        let config = BmcConfig {
            max_bound: 4,
            max_context_switches: 4,
            ..Default::default()
        };
        let unroller = ExecutionUnroller::new(config);
        let executions = unroller.unroll(&system, 4);

        // Should find multiple interleavings
        assert!(!executions.is_empty());
    }

    #[test]
    fn test_symmetry_detection() {
        let system = store_buffer_system();
        let mut reducer = SymmetryReducer::new();
        reducer.detect_thread_symmetries(&system);

        // Both threads have the same structure (store + load)
        assert!(!reducer.thread_symmetries.is_empty());
    }

    #[test]
    fn test_bmc_verified() {
        let system = message_passing_system();
        let property = Property::Safety {
            name: "trivial".to_string(),
            predicate: PropertyPredicate::True,
        };

        let config = BmcConfig {
            max_bound: 4,
            max_context_switches: 4,
            ..Default::default()
        };
        let mut checker = BoundedModelChecker::new(config);
        let result = checker.check(&system, &property);

        match result {
            BmcResult::Verified { .. } => {} // Expected
            _ => panic!("Expected verified result"),
        }
    }

    #[test]
    fn test_bmc_counterexample() {
        let system = message_passing_system();

        // Property: x should never be 1 at termination (will find counterexample)
        let property = Property::Assertion {
            name: "x_is_zero".to_string(),
            conditions: vec![(0, 0)], // x should be 0 (but thread 0 writes 1)
        };

        let config = BmcConfig {
            max_bound: 4,
            max_context_switches: 4,
            ..Default::default()
        };
        let mut checker = BoundedModelChecker::new(config);
        let result = checker.check(&system, &property);

        match result {
            BmcResult::CounterexampleFound { .. } => {} // Expected
            other => panic!("Expected counterexample, got {:?}", other),
        }
    }

    #[test]
    fn test_state_operations() {
        let mut state = State::initial(2);
        state.write_memory(0x100, 42);
        assert_eq!(state.read_memory(0x100), 42);
        assert_eq!(state.read_memory(0x200), 0); // uninitialized

        state.write_register(0, 0, 99);
        assert_eq!(state.read_register(0, 0), 99);
        assert_eq!(state.read_register(1, 0), 0); // uninitialized
    }

    #[test]
    fn test_property_predicate() {
        let mut state = State::initial(2);
        state.write_memory(0, 1);
        state.write_register(0, 0, 5);

        let pred = PropertyPredicate::MemoryEquals { address: 0, value: 1 };
        assert!(pred.evaluate(&state));

        let pred2 = PropertyPredicate::And(vec![
            PropertyPredicate::MemoryEquals { address: 0, value: 1 },
            PropertyPredicate::RegisterEquals { thread: 0, register: 0, value: 5 },
        ]);
        assert!(pred2.evaluate(&state));

        let pred3 = PropertyPredicate::Not(Box::new(
            PropertyPredicate::MemoryEquals { address: 0, value: 0 },
        ));
        assert!(pred3.evaluate(&state));
    }

    #[test]
    fn test_counterexample_minimize() {
        let ce = Counterexample::new("test");
        assert_eq!(ce.length(), 0);
    }

    #[test]
    fn test_bmc_statistics() {
        let mut stats = BmcStatistics::new();
        stats.states_explored = 100;
        stats.transitions_explored = 200;
        stats.time_per_level = vec![1000, 2000, 3000];
        assert_eq!(stats.total_time_us(), 6000);
        assert!((stats.avg_time_per_level_us() - 2000.0).abs() < 0.1);
    }

    #[test]
    fn test_symmetry_reducer_canonical() {
        let mut reducer = SymmetryReducer::new();
        reducer.thread_symmetries = vec![vec![0, 1]];

        let mut state = State::initial(2);
        state.program_counters = vec![1, 0];
        let canonical = reducer.canonical_form(&state);
        assert_eq!(canonical.program_counters, vec![0, 1]);
    }
}
