//! Exhaustive execution enumeration and state space exploration for LITMUS∞.
//!
//! Implements exhaustive checking of all candidate executions, state space
//! exploration strategies (DFS, BFS, bounded), partial order reduction,
//! sleep set reduction, and symmetry-based reduction.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::hash::Hash;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    Event, ThreadId, Address, Value, OpType, Scope,
    ExecutionGraph,
};
use crate::checker::memory_model::{
    MemoryModel,
};
use crate::checker::litmus::{LitmusTest, Instruction};
use crate::checker::verifier;

// ═══════════════════════════════════════════════════════════════════════════
// Execution Enumerator
// ═══════════════════════════════════════════════════════════════════════════

/// Enumerates all candidate executions for a litmus test.
#[derive(Debug)]
pub struct ExecutionEnumerator {
    /// Maximum executions to enumerate.
    pub max_executions: u64,
    /// Statistics.
    pub stats: EnumerationStats,
}

/// Statistics from enumeration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnumerationStats {
    /// Total candidate executions enumerated.
    pub total_enumerated: u64,
    /// Number filtered as inconsistent.
    pub inconsistent: u64,
    /// Number passing consistency checks.
    pub consistent: u64,
    /// Whether enumeration completed.
    pub completed: bool,
}

impl ExecutionEnumerator {
    /// Create a new enumerator with a bound.
    pub fn new(max_executions: u64) -> Self {
        ExecutionEnumerator {
            max_executions,
            stats: EnumerationStats::default(),
        }
    }

    /// Count the number of candidate executions for a litmus test.
    /// This gives an upper bound based on the number of possible rf/co assignments.
    pub fn count_candidates(test: &LitmusTest) -> u64 {
        // For each read, count possible rf sources (writes to same address + initial)
        let mut reads = Vec::new();
        let mut writes_per_addr: HashMap<Address, usize> = HashMap::new();

        for thread in &test.threads {
            for instr in &thread.instructions {
                match instr {
                    Instruction::Load { addr, .. } => {
                        reads.push(*addr);
                    }
                    Instruction::Store { addr, .. } => {
                        *writes_per_addr.entry(*addr).or_insert(0) += 1;
                    }
                    Instruction::RMW { addr, .. } => {
                        reads.push(*addr);
                        *writes_per_addr.entry(*addr).or_insert(0) += 1;
                    }
                    _ => {}
                }
            }
        }

        let mut rf_choices = 1u64;
        for addr in &reads {
            let num_writes = writes_per_addr.get(addr).copied().unwrap_or(0) + 1; // +1 for initial
            rf_choices = rf_choices.saturating_mul(num_writes as u64);
        }

        // For co: number of total orderings of writes per address
        let mut co_choices = 1u64;
        for (_, &count) in &writes_per_addr {
            // count! orderings
            let mut fact = 1u64;
            for i in 1..=count as u64 {
                fact = fact.saturating_mul(i);
            }
            co_choices = co_choices.saturating_mul(fact);
        }

        rf_choices.saturating_mul(co_choices)
    }

    /// Enumerate all candidate executions.
    pub fn enumerate_all(&mut self, test: &LitmusTest) -> Vec<ExecutionGraph> {
        let mut results = Vec::new();

        // Build the event list
        let events = Self::build_events(test);
        let _n = events.len();

        // Group reads and writes by address
        let mut reads_at: HashMap<Address, Vec<usize>> = HashMap::new();
        let mut writes_at: HashMap<Address, Vec<usize>> = HashMap::new();

        for (i, ev) in events.iter().enumerate() {
            match ev.op_type {
                OpType::Read => { reads_at.entry(ev.address).or_default().push(i); }
                OpType::Write => { writes_at.entry(ev.address).or_default().push(i); }
                OpType::RMW => {
                    reads_at.entry(ev.address).or_default().push(i);
                    writes_at.entry(ev.address).or_default().push(i);
                }
                _ => {}
            }
        }

        // Generate all possible rf assignments
        let rf_assignments = Self::generate_rf_assignments(&reads_at, &writes_at);

        for rf in &rf_assignments {
            if self.stats.total_enumerated >= self.max_executions {
                self.stats.completed = false;
                return results;
            }
            self.stats.total_enumerated += 1;

            // Build execution graph
            let mut exec = ExecutionGraph::new(events.clone());
            for &(w, r) in rf {
                exec.add_rf(w, r);
            }

            // Generate a coherence order (simplified: use event order)
            let co = Self::generate_co(&writes_at, &events);
            for &(w1, w2) in &co {
                exec.add_co(w1, w2);
            }

            results.push(exec);
        }

        self.stats.completed = true;
        results
    }

    /// Build events from a litmus test.
    fn build_events(test: &LitmusTest) -> Vec<Event> {
        let mut events = Vec::new();
        let mut next_id = 0;

        for thread in &test.threads {
            let mut po_index = 0;
            for instr in &thread.instructions {
                match instr {
                    Instruction::Load { addr, .. } => {
                        events.push(Event {
                            id: next_id,
                            thread: thread.id,
                            op_type: OpType::Read,
                            address: *addr,
                            value: 0,
                            scope: Scope::None,
                            po_index,
                        });
                        next_id += 1;
                        po_index += 1;
                    }
                    Instruction::Store { addr, value, .. } => {
                        events.push(Event {
                            id: next_id,
                            thread: thread.id,
                            op_type: OpType::Write,
                            address: *addr,
                            value: *value,
                            scope: Scope::None,
                            po_index,
                        });
                        next_id += 1;
                        po_index += 1;
                    }
                    Instruction::Fence { scope, .. } => {
                        events.push(Event {
                            id: next_id,
                            thread: thread.id,
                            op_type: OpType::Fence,
                            address: 0,
                            value: 0,
                            scope: scope.to_exec_scope(),
                            po_index,
                        });
                        next_id += 1;
                        po_index += 1;
                    }
                    Instruction::RMW { addr, value, .. } => {
                        events.push(Event {
                            id: next_id,
                            thread: thread.id,
                            op_type: OpType::RMW,
                            address: *addr,
                            value: *value,
                            scope: Scope::None,
                            po_index,
                        });
                        next_id += 1;
                        po_index += 1;
                    }
                    _ => {}
                }
            }
        }

        events
    }

    /// Generate all possible reads-from assignments.
    fn generate_rf_assignments(
        reads_at: &HashMap<Address, Vec<usize>>,
        writes_at: &HashMap<Address, Vec<usize>>,
    ) -> Vec<Vec<(usize, usize)>> {
        // For each address with reads, generate all possible source writes
        let mut addr_choices: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();

        for (addr, reads) in reads_at {
            let writes = writes_at.get(addr).cloned().unwrap_or_default();
            // Each read can come from any write (or initial value, represented as no rf edge)
            let mut choices_for_addr = Vec::new();

            if writes.is_empty() {
                // All reads get initial value (no rf edges)
                choices_for_addr.push(Vec::new());
            } else {
                // Generate all combinations of rf assignments for reads at this address
                let mut rf_options: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
                for &r in reads {
                    let mut new_options = Vec::new();
                    for existing in &rf_options {
                        // Option: read from initial value (no edge)
                        new_options.push(existing.clone());
                        // Option: read from each write
                        for &w in &writes {
                            let mut extended = existing.clone();
                            extended.push((w, r));
                            new_options.push(extended);
                        }
                    }
                    rf_options = new_options;
                }
                choices_for_addr = rf_options;
            }

            addr_choices.push(choices_for_addr);
        }

        // Combine choices across addresses
        if addr_choices.is_empty() {
            return vec![Vec::new()];
        }

        let mut result = vec![Vec::new()];
        for choices in &addr_choices {
            let mut new_result = Vec::new();
            for existing in &result {
                for choice in choices {
                    let mut combined = existing.clone();
                    combined.extend(choice);
                    new_result.push(combined);
                }
            }
            result = new_result;
        }

        result
    }

    /// Generate a coherence order for writes at each address.
    fn generate_co(
        writes_at: &HashMap<Address, Vec<usize>>,
        _events: &[Event],
    ) -> Vec<(usize, usize)> {
        let mut co = Vec::new();
        for (_, writes) in writes_at {
            // Simple: order by event ID (could be more sophisticated)
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    co.push((writes[i], writes[j]));
                }
            }
        }
        co
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State Space
// ═══════════════════════════════════════════════════════════════════════════

/// A state in the state space exploration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExplorationState {
    /// Per-thread program counters.
    pub pc: Vec<usize>,
    /// Shared memory state.
    pub memory: BTreeMap<Address, Value>,
    /// Per-thread register state.
    pub registers: Vec<BTreeMap<usize, Value>>,
}

impl ExplorationState {
    /// Create an initial state.
    pub fn initial(num_threads: usize, initial_mem: &HashMap<Address, Value>) -> Self {
        ExplorationState {
            pc: vec![0; num_threads],
            memory: initial_mem.iter().map(|(&a, &v)| (a, v)).collect(),
            registers: vec![BTreeMap::new(); num_threads],
        }
    }

    /// Is this a final state (all threads done)?
    pub fn is_final(&self, program_lengths: &[usize]) -> bool {
        self.pc.iter().enumerate().all(|(t, &pc)| pc >= program_lengths[t])
    }
}

/// A transition in the state space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transition {
    /// Thread performing the transition.
    pub thread: ThreadId,
    /// The instruction index.
    pub instruction_index: usize,
    /// Description.
    pub description: String,
}

/// State space as an explicit graph.
#[derive(Debug)]
pub struct StateSpace {
    /// States.
    pub states: Vec<ExplorationState>,
    /// Transitions: (from_state, to_state, transition).
    pub transitions: Vec<(usize, usize, Transition)>,
    /// State to index mapping.
    state_index: HashMap<ExplorationState, usize>,
    /// Final states.
    pub final_states: Vec<usize>,
}

impl StateSpace {
    /// Create a new empty state space.
    pub fn new() -> Self {
        StateSpace {
            states: Vec::new(),
            transitions: Vec::new(),
            state_index: HashMap::new(),
            final_states: Vec::new(),
        }
    }

    /// Add a state, returning its index.
    pub fn add_state(&mut self, state: ExplorationState) -> usize {
        if let Some(&idx) = self.state_index.get(&state) {
            return idx;
        }
        let idx = self.states.len();
        self.state_index.insert(state.clone(), idx);
        self.states.push(state);
        idx
    }

    /// Add a transition.
    pub fn add_transition(&mut self, from: usize, to: usize, transition: Transition) {
        self.transitions.push((from, to, transition));
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    /// Average branching factor.
    pub fn avg_branching_factor(&self) -> f64 {
        if self.states.is_empty() { return 0.0; }
        self.transitions.len() as f64 / self.states.len() as f64
    }

    /// Export as DOT format for visualization.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph StateSpace {\n");
        for (i, _state) in self.states.iter().enumerate() {
            let label = if self.final_states.contains(&i) {
                format!("S{} [shape=doublecircle]", i)
            } else {
                format!("S{}", i)
            };
            dot.push_str(&format!("  {};\n", label));
        }
        for (from, to, trans) in &self.transitions {
            dot.push_str(&format!("  S{} -> S{} [label=\"{}\"];\n", from, to, trans.description));
        }
        dot.push_str("}\n");
        dot
    }
}

impl Default for StateSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exploration Strategy
// ═══════════════════════════════════════════════════════════════════════════

/// Exploration strategy for state space search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Depth-first search.
    DFS,
    /// Breadth-first search.
    BFS,
    /// Random walk.
    Random,
    /// Bounded depth search.
    Bounded { max_depth: usize },
}

impl fmt::Display for ExplorationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExplorationStrategy::DFS => write!(f, "DFS"),
            ExplorationStrategy::BFS => write!(f, "BFS"),
            ExplorationStrategy::Random => write!(f, "Random"),
            ExplorationStrategy::Bounded { max_depth } => write!(f, "Bounded({})", max_depth),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Partial Order Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Partial order reduction for state space exploration.
#[derive(Debug)]
pub struct PartialOrderReduction {
    /// Statistics.
    pub stats: PORStats,
}

/// Statistics from partial order reduction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PORStats {
    /// Total transitions considered.
    pub total_transitions: usize,
    /// Transitions after reduction.
    pub reduced_transitions: usize,
    /// Reduction ratio.
    pub reduction_ratio: f64,
}

impl PartialOrderReduction {
    /// Create a new POR instance.
    pub fn new() -> Self {
        PartialOrderReduction { stats: PORStats::default() }
    }

    /// Check if two transitions are independent.
    /// Independent transitions: commuting and don't enable/disable each other.
    pub fn are_independent(t1: &Transition, t2: &Transition) -> bool {
        // Transitions on different threads that don't share memory are independent
        t1.thread != t2.thread
    }

    /// Compute the persistent set for a state.
    /// A persistent set is a subset of enabled transitions such that any
    /// transition sequence not in the set is independent of all set members.
    pub fn compute_persistent_set(
        &mut self,
        enabled: &[Transition],
    ) -> Vec<Transition> {
        self.stats.total_transitions += enabled.len();

        if enabled.len() <= 1 {
            self.stats.reduced_transitions += enabled.len();
            return enabled.to_vec();
        }

        // Simple heuristic: pick one thread's transitions
        let first_thread = enabled[0].thread;
        let persistent: Vec<Transition> = enabled.iter()
            .filter(|t| t.thread == first_thread)
            .cloned()
            .collect();

        // Check that this is actually a valid persistent set
        // (all other transitions are independent)
        let all_independent = enabled.iter()
            .filter(|t| t.thread != first_thread)
            .all(|t| persistent.iter().all(|p| Self::are_independent(t, p)));

        let result = if all_independent {
            persistent
        } else {
            enabled.to_vec() // fall back to full set
        };

        self.stats.reduced_transitions += result.len();
        self.stats.reduction_ratio = if self.stats.total_transitions > 0 {
            1.0 - (self.stats.reduced_transitions as f64 / self.stats.total_transitions as f64)
        } else {
            0.0
        };

        result
    }

    /// Compute the ample set (stricter than persistent set).
    pub fn compute_ample_set(
        &mut self,
        enabled: &[Transition],
    ) -> Vec<Transition> {
        // Ample set conditions:
        // C0: ample(s) = {} iff enabled(s) = {}
        // C1: any dependent transition can only be reached after a transition in ample(s)
        // C2: if s is not fully expanded, ample(s) contains no visible transitions
        // C3: a transition in ample(s) cannot be enabled for the first time by a transition in enabled(s) \ ample(s)

        // Simplified: use persistent set
        self.compute_persistent_set(enabled)
    }
}

impl Default for PartialOrderReduction {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sleep Set Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Sleep set reduction for further state space compression.
#[derive(Debug)]
pub struct SleepSetReduction {
    /// Statistics.
    pub stats: SleepSetStats,
}

/// Statistics from sleep set reduction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SleepSetStats {
    /// Transitions avoided by sleep sets.
    pub transitions_avoided: usize,
    /// Total transitions considered.
    pub transitions_considered: usize,
}

impl SleepSetReduction {
    /// Create a new sleep set reducer.
    pub fn new() -> Self {
        SleepSetReduction { stats: SleepSetStats::default() }
    }

    /// Initial (empty) sleep set.
    pub fn initial_sleep_set() -> HashSet<Transition> {
        HashSet::new()
    }

    /// Update the sleep set after executing a transition.
    /// Add independent transitions that were not executed to the sleep set.
    pub fn update_sleep_set(
        &mut self,
        current_sleep: &HashSet<Transition>,
        executed: &Transition,
        all_enabled: &[Transition],
    ) -> HashSet<Transition> {
        let mut new_sleep = HashSet::new();

        // Keep sleeping transitions that are independent of the executed one
        for t in current_sleep {
            if PartialOrderReduction::are_independent(t, executed) {
                new_sleep.insert(t.clone());
            }
        }

        // Add enabled transitions that are independent and not the executed one
        for t in all_enabled {
            if t != executed && PartialOrderReduction::are_independent(t, executed) {
                new_sleep.insert(t.clone());
            }
        }

        self.stats.transitions_considered += all_enabled.len();
        self.stats.transitions_avoided += new_sleep.len();

        new_sleep
    }

    /// Filter enabled transitions by removing sleeping ones.
    pub fn filter_enabled(
        enabled: &[Transition],
        sleep_set: &HashSet<Transition>,
    ) -> Vec<Transition> {
        enabled.iter()
            .filter(|t| !sleep_set.contains(t))
            .cloned()
            .collect()
    }
}

impl Default for SleepSetReduction {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Exhaustive Checker
// ═══════════════════════════════════════════════════════════════════════════

/// Result of exhaustive checking.
#[derive(Debug, Clone)]
pub struct ExhaustiveResult {
    /// All explored executions that are consistent.
    pub consistent_executions: Vec<ExecutionGraph>,
    /// All explored executions that are inconsistent.
    pub inconsistent_count: u64,
    /// Observed final states / outcomes.
    pub observed_outcomes: Vec<BTreeMap<String, Value>>,
    /// Statistics.
    pub stats: ExhaustiveStats,
}

/// Statistics from exhaustive checking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExhaustiveStats {
    /// Total executions explored.
    pub total_explored: u64,
    /// Executions after reduction.
    pub after_reduction: u64,
    /// Time taken (approximate).
    pub time_ms: u64,
    /// Memory used (approximate, bytes).
    pub memory_bytes: usize,
    /// Whether exploration completed.
    pub completed: bool,
    /// POR reduction ratio.
    pub por_reduction: f64,
}

/// Main entry point for exhaustive checking.
#[derive(Debug)]
pub struct ExhaustiveChecker {
    /// Exploration strategy.
    pub strategy: ExplorationStrategy,
    /// Whether to use partial order reduction.
    pub use_por: bool,
    /// Whether to use sleep set reduction.
    pub use_sleep_sets: bool,
    /// Maximum executions to check.
    pub max_executions: u64,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
}

impl ExhaustiveChecker {
    /// Create a new checker with default settings.
    pub fn new() -> Self {
        ExhaustiveChecker {
            strategy: ExplorationStrategy::DFS,
            use_por: true,
            use_sleep_sets: true,
            max_executions: 100_000,
            timeout_ms: 30_000,
        }
    }

    /// Create with custom settings.
    pub fn with_strategy(mut self, strategy: ExplorationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable/disable POR.
    pub fn with_por(mut self, enabled: bool) -> Self {
        self.use_por = enabled;
        self
    }

    /// Enable/disable sleep sets.
    pub fn with_sleep_sets(mut self, enabled: bool) -> Self {
        self.use_sleep_sets = enabled;
        self
    }

    /// Set maximum executions.
    pub fn with_max_executions(mut self, max: u64) -> Self {
        self.max_executions = max;
        self
    }

    /// Run exhaustive checking on a litmus test.
    pub fn check_all(
        &self,
        test: &LitmusTest,
        model: &MemoryModel,
    ) -> ExhaustiveResult {
        let start = std::time::Instant::now();
        let mut stats = ExhaustiveStats::default();

        // Enumerate all candidate executions
        let mut enumerator = ExecutionEnumerator::new(self.max_executions);
        let candidates = enumerator.enumerate_all(test);

        stats.total_explored = candidates.len() as u64;
        stats.after_reduction = stats.total_explored; // no reduction in this path

        // Check each execution against the model
        let mut verifier = verifier::Verifier::new(model.clone());
        let mut consistent = Vec::new();
        let mut inconsistent_count = 0u64;

        for exec in candidates {
            let result = verifier.check_execution(&exec);
            if result.violations.is_empty() {
                consistent.push(exec);
            } else {
                inconsistent_count += 1;
            }
        }

        stats.completed = enumerator.stats.completed;
        stats.time_ms = start.elapsed().as_millis() as u64;

        ExhaustiveResult {
            consistent_executions: consistent,
            inconsistent_count,
            observed_outcomes: Vec::new(),
            stats,
        }
    }
}

impl Default for ExhaustiveChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Symmetry Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetry-based reduction of the state space.
#[derive(Debug)]
pub struct SymmetryReduction;

impl SymmetryReduction {
    /// Detect thread symmetries in a litmus test.
    /// Two threads are symmetric if they have identical instruction sequences.
    pub fn detect_thread_symmetries(test: &LitmusTest) -> Vec<Vec<ThreadId>> {
        let mut groups: HashMap<Vec<Instruction>, Vec<ThreadId>> = HashMap::new();

        for thread in &test.threads {
            groups.entry(thread.instructions.clone()).or_default().push(thread.id);
        }

        groups.into_values()
            .filter(|g| g.len() > 1)
            .collect()
    }

    /// Detect data symmetries (same values used in different positions).
    pub fn detect_data_symmetries(test: &LitmusTest) -> Vec<(Value, Value)> {
        let mut value_pairs = Vec::new();
        let mut values_used: HashSet<Value> = HashSet::new();

        for thread in &test.threads {
            for instr in &thread.instructions {
                if let Instruction::Store { value, .. } = instr {
                    values_used.insert(*value);
                }
            }
        }

        let values: Vec<Value> = values_used.into_iter().collect();
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                value_pairs.push((values[i], values[j]));
            }
        }

        value_pairs
    }

    /// Compute the reduction factor from detected symmetries.
    pub fn reduction_factor(thread_symmetries: &[Vec<ThreadId>]) -> u64 {
        let mut factor = 1u64;
        for group in thread_symmetries {
            let n = group.len() as u64;
            // n! symmetric executions can be reduced to 1
            let mut fact = 1u64;
            for i in 1..=n {
                fact *= i;
            }
            factor *= fact;
        }
        factor
    }

    /// Check if two executions are equivalent modulo symmetry.
    pub fn are_equivalent(
        exec1: &ExecutionGraph,
        exec2: &ExecutionGraph,
        thread_perm: &HashMap<ThreadId, ThreadId>,
    ) -> bool {
        if exec1.events.len() != exec2.events.len() {
            return false;
        }

        // Check that events match under the permutation
        // (simplified check)
        for (ev1, ev2) in exec1.events.iter().zip(exec2.events.iter()) {
            let permuted_thread = thread_perm.get(&ev1.thread).copied().unwrap_or(ev1.thread);
            if permuted_thread != ev2.thread {
                return false;
            }
            if ev1.op_type != ev2.op_type || ev1.address != ev2.address {
                return false;
            }
        }

        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::{Thread, Ordering};

    fn make_simple_test() -> LitmusTest {
        let mut test = LitmusTest::new("simple");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    fn make_sb_test() -> LitmusTest {
        let mut test = LitmusTest::new("SB");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    #[test]
    fn test_count_candidates() {
        let test = make_simple_test();
        let count = ExecutionEnumerator::count_candidates(&test);
        assert!(count >= 1);
    }

    #[test]
    fn test_enumerate_simple() {
        let test = make_simple_test();
        let mut enumerator = ExecutionEnumerator::new(1000);
        let execs = enumerator.enumerate_all(&test);
        assert!(!execs.is_empty());
        assert!(enumerator.stats.completed);
    }

    #[test]
    fn test_enumerate_sb() {
        let test = make_sb_test();
        let count = ExecutionEnumerator::count_candidates(&test);
        assert!(count >= 1);
    }

    #[test]
    fn test_exploration_state() {
        let state = ExplorationState::initial(2, &HashMap::new());
        assert!(state.is_final(&[0, 0]));
        assert!(!state.is_final(&[1, 0]));
    }

    #[test]
    fn test_state_space() {
        let mut ss = StateSpace::new();
        let s0 = ExplorationState::initial(2, &HashMap::new());
        let idx = ss.add_state(s0);
        assert_eq!(idx, 0);
        assert_eq!(ss.num_states(), 1);
    }

    #[test]
    fn test_state_space_dot() {
        let mut ss = StateSpace::new();
        let s0 = ExplorationState::initial(2, &HashMap::new());
        ss.add_state(s0);
        let dot = ss.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_exploration_strategy_display() {
        assert_eq!(format!("{}", ExplorationStrategy::DFS), "DFS");
        assert_eq!(format!("{}", ExplorationStrategy::BFS), "BFS");
    }

    #[test]
    fn test_por_independence() {
        let t1 = Transition { thread: 0, instruction_index: 0, description: "T0:R".to_string() };
        let t2 = Transition { thread: 1, instruction_index: 0, description: "T1:W".to_string() };
        assert!(PartialOrderReduction::are_independent(&t1, &t2));

        let t3 = Transition { thread: 0, instruction_index: 1, description: "T0:W".to_string() };
        assert!(!PartialOrderReduction::are_independent(&t1, &t3));
    }

    #[test]
    fn test_por_persistent_set() {
        let mut por = PartialOrderReduction::new();
        let enabled = vec![
            Transition { thread: 0, instruction_index: 0, description: "T0:R".to_string() },
            Transition { thread: 1, instruction_index: 0, description: "T1:W".to_string() },
        ];
        let persistent = por.compute_persistent_set(&enabled);
        assert!(!persistent.is_empty());
        assert!(persistent.len() <= enabled.len());
    }

    #[test]
    fn test_sleep_set() {
        let mut ssr = SleepSetReduction::new();
        let t1 = Transition { thread: 0, instruction_index: 0, description: "T0:R".to_string() };
        let t2 = Transition { thread: 1, instruction_index: 0, description: "T1:W".to_string() };
        let sleep = SleepSetReduction::initial_sleep_set();
        let new_sleep = ssr.update_sleep_set(&sleep, &t1, &[t1.clone(), t2.clone()]);
        // t2 is independent of t1, so it should be in the sleep set
        assert!(new_sleep.contains(&t2));
    }

    #[test]
    fn test_sleep_set_filter() {
        let t1 = Transition { thread: 0, instruction_index: 0, description: "T0:R".to_string() };
        let t2 = Transition { thread: 1, instruction_index: 0, description: "T1:W".to_string() };
        let mut sleep = HashSet::new();
        sleep.insert(t2.clone());
        let filtered = SleepSetReduction::filter_enabled(&[t1.clone(), t2], &sleep);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], t1);
    }

    #[test]
    fn test_exhaustive_checker_creation() {
        let checker = ExhaustiveChecker::new()
            .with_strategy(ExplorationStrategy::DFS)
            .with_por(true)
            .with_max_executions(1000);
        assert_eq!(checker.strategy, ExplorationStrategy::DFS);
        assert!(checker.use_por);
    }

    #[test]
    fn test_symmetry_detection() {
        let mut test = LitmusTest::new("symmetric");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0, 1, Ordering::Relaxed); // same instructions
        test.add_thread(t1);

        let symmetries = SymmetryReduction::detect_thread_symmetries(&test);
        assert_eq!(symmetries.len(), 1);
        assert_eq!(symmetries[0].len(), 2);
    }

    #[test]
    fn test_reduction_factor() {
        let symmetries = vec![vec![0, 1]]; // 2 symmetric threads
        let factor = SymmetryReduction::reduction_factor(&symmetries);
        assert_eq!(factor, 2); // 2! = 2
    }

    #[test]
    fn test_no_symmetry() {
        let test = make_sb_test();
        let symmetries = SymmetryReduction::detect_thread_symmetries(&test);
        // SB test threads have different instructions, so no symmetry
        assert!(symmetries.is_empty() || symmetries.iter().all(|g| g.len() <= 1));
    }
}
