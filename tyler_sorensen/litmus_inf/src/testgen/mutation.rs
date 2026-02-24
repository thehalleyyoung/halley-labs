//! Litmus test mutation operators for LITMUS∞.
//!
//! Provides a comprehensive set of mutation operators for transforming
//! litmus tests: instruction mutation, ordering mutation, thread reordering,
//! fence insertion/removal, and memory location mutation.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::checker::{
    LitmusTest, Thread, Instruction, Outcome, LitmusOutcome,
    Address, Value,
};
use crate::checker::litmus::{Ordering, Scope, RegId};

// ---------------------------------------------------------------------------
// MutationOperator — the set of mutations
// ---------------------------------------------------------------------------

/// A single mutation operator applicable to a litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MutationOperator {
    /// Replace the ordering on a specific instruction.
    ChangeOrdering {
        thread: usize,
        instr: usize,
        new_ordering: Ordering,
    },
    /// Swap a load for a store (or vice versa).
    SwapLoadStore {
        thread: usize,
        instr: usize,
    },
    /// Insert a fence after a given instruction.
    InsertFence {
        thread: usize,
        position: usize,
        ordering: Ordering,
        scope: Scope,
    },
    /// Remove a fence instruction.
    RemoveFence {
        thread: usize,
        instr: usize,
    },
    /// Change the memory address of an instruction.
    ChangeAddress {
        thread: usize,
        instr: usize,
        new_addr: Address,
    },
    /// Change the stored value.
    ChangeValue {
        thread: usize,
        instr: usize,
        new_value: Value,
    },
    /// Swap the order of two instructions in a thread.
    SwapInstructions {
        thread: usize,
        instr_a: usize,
        instr_b: usize,
    },
    /// Duplicate an instruction.
    DuplicateInstruction {
        thread: usize,
        instr: usize,
    },
    /// Remove an instruction.
    RemoveInstruction {
        thread: usize,
        instr: usize,
    },
    /// Swap two entire threads.
    SwapThreads {
        thread_a: usize,
        thread_b: usize,
    },
    /// Add a new thread with given instructions.
    AddThread {
        instructions: Vec<Instruction>,
    },
    /// Remove a thread.
    RemoveThread {
        thread: usize,
    },
    /// Change a load into an RMW.
    LoadToRMW {
        thread: usize,
        instr: usize,
        write_value: Value,
    },
    /// Change an RMW into a load.
    RMWToLoad {
        thread: usize,
        instr: usize,
    },
    /// Strengthen ordering (Relaxed → Acquire/Release → SeqCst).
    StrengthenOrdering {
        thread: usize,
        instr: usize,
    },
    /// Weaken ordering (SeqCst → Acquire/Release → Relaxed).
    WeakenOrdering {
        thread: usize,
        instr: usize,
    },
    /// Change fence scope.
    ChangeFenceScope {
        thread: usize,
        instr: usize,
        new_scope: Scope,
    },
}

impl fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChangeOrdering { thread, instr, new_ordering } =>
                write!(f, "ChangeOrdering(T{}[{}] → {})", thread, instr, new_ordering),
            Self::SwapLoadStore { thread, instr } =>
                write!(f, "SwapLoadStore(T{}[{}])", thread, instr),
            Self::InsertFence { thread, position, ordering, scope } =>
                write!(f, "InsertFence(T{}@{}, {}{}", thread, position, ordering, scope),
            Self::RemoveFence { thread, instr } =>
                write!(f, "RemoveFence(T{}[{}])", thread, instr),
            Self::ChangeAddress { thread, instr, new_addr } =>
                write!(f, "ChangeAddress(T{}[{}] → {:#x})", thread, instr, new_addr),
            Self::ChangeValue { thread, instr, new_value } =>
                write!(f, "ChangeValue(T{}[{}] → {})", thread, instr, new_value),
            Self::SwapInstructions { thread, instr_a, instr_b } =>
                write!(f, "SwapInstructions(T{}[{},{}])", thread, instr_a, instr_b),
            Self::DuplicateInstruction { thread, instr } =>
                write!(f, "DuplicateInstr(T{}[{}])", thread, instr),
            Self::RemoveInstruction { thread, instr } =>
                write!(f, "RemoveInstr(T{}[{}])", thread, instr),
            Self::SwapThreads { thread_a, thread_b } =>
                write!(f, "SwapThreads(T{},T{})", thread_a, thread_b),
            Self::AddThread { instructions } =>
                write!(f, "AddThread({} instrs)", instructions.len()),
            Self::RemoveThread { thread } =>
                write!(f, "RemoveThread(T{})", thread),
            Self::LoadToRMW { thread, instr, write_value } =>
                write!(f, "LoadToRMW(T{}[{}], val={})", thread, instr, write_value),
            Self::RMWToLoad { thread, instr } =>
                write!(f, "RMWToLoad(T{}[{}])", thread, instr),
            Self::StrengthenOrdering { thread, instr } =>
                write!(f, "StrengthenOrdering(T{}[{}])", thread, instr),
            Self::WeakenOrdering { thread, instr } =>
                write!(f, "WeakenOrdering(T{}[{}])", thread, instr),
            Self::ChangeFenceScope { thread, instr, new_scope } =>
                write!(f, "ChangeFenceScope(T{}[{}] → {})", thread, instr, new_scope),
        }
    }
}

impl MutationOperator {
    /// Check whether this mutation is applicable to the given test.
    pub fn applicable(&self, test: &LitmusTest) -> bool {
        match self {
            Self::ChangeOrdering { thread, instr, .. } |
            Self::SwapLoadStore { thread, instr } |
            Self::ChangeAddress { thread, instr, .. } |
            Self::ChangeValue { thread, instr, .. } |
            Self::DuplicateInstruction { thread, instr } |
            Self::RemoveInstruction { thread, instr } |
            Self::StrengthenOrdering { thread, instr } |
            Self::WeakenOrdering { thread, instr } => {
                *thread < test.threads.len() && *instr < test.threads[*thread].instructions.len()
            }
            Self::InsertFence { thread, position, .. } => {
                *thread < test.threads.len() && *position <= test.threads[*thread].instructions.len()
            }
            Self::RemoveFence { thread, instr } => {
                *thread < test.threads.len()
                    && *instr < test.threads[*thread].instructions.len()
                    && matches!(test.threads[*thread].instructions[*instr], Instruction::Fence { .. })
            }
            Self::SwapInstructions { thread, instr_a, instr_b } => {
                *thread < test.threads.len()
                    && *instr_a < test.threads[*thread].instructions.len()
                    && *instr_b < test.threads[*thread].instructions.len()
                    && instr_a != instr_b
            }
            Self::SwapThreads { thread_a, thread_b } => {
                *thread_a < test.threads.len()
                    && *thread_b < test.threads.len()
                    && thread_a != thread_b
            }
            Self::AddThread { .. } => true,
            Self::RemoveThread { thread } => {
                *thread < test.threads.len() && test.threads.len() > 1
            }
            Self::LoadToRMW { thread, instr, .. } => {
                *thread < test.threads.len()
                    && *instr < test.threads[*thread].instructions.len()
                    && matches!(test.threads[*thread].instructions[*instr], Instruction::Load { .. })
            }
            Self::RMWToLoad { thread, instr } => {
                *thread < test.threads.len()
                    && *instr < test.threads[*thread].instructions.len()
                    && matches!(test.threads[*thread].instructions[*instr], Instruction::RMW { .. })
            }
            Self::ChangeFenceScope { thread, instr, .. } => {
                *thread < test.threads.len()
                    && *instr < test.threads[*thread].instructions.len()
                    && matches!(test.threads[*thread].instructions[*instr], Instruction::Fence { .. })
            }
        }
    }

    /// Apply this mutation to a test, returning the mutated copy.
    pub fn apply(&self, test: &LitmusTest) -> Option<LitmusTest> {
        if !self.applicable(test) { return None; }

        let mut mutant = test.clone();
        match self {
            Self::ChangeOrdering { thread, instr, new_ordering } => {
                set_ordering(&mut mutant.threads[*thread].instructions[*instr], *new_ordering);
            }
            Self::SwapLoadStore { thread, instr } => {
                let i = &mutant.threads[*thread].instructions[*instr];
                let new_instr = match i {
                    Instruction::Load { reg: _, addr, ordering } => {
                        Instruction::Store { addr: *addr, value: 1, ordering: *ordering }
                    }
                    Instruction::Store { addr, value: _, ordering } => {
                        Instruction::Load { reg: 0, addr: *addr, ordering: *ordering }
                    }
                    _ => return None,
                };
                mutant.threads[*thread].instructions[*instr] = new_instr;
            }
            Self::InsertFence { thread, position, ordering, scope } => {
                mutant.threads[*thread].instructions.insert(
                    *position,
                    Instruction::Fence { ordering: *ordering, scope: *scope },
                );
            }
            Self::RemoveFence { thread, instr } => {
                mutant.threads[*thread].instructions.remove(*instr);
            }
            Self::ChangeAddress { thread, instr, new_addr } => {
                set_address(&mut mutant.threads[*thread].instructions[*instr], *new_addr);
            }
            Self::ChangeValue { thread, instr, new_value } => {
                set_value(&mut mutant.threads[*thread].instructions[*instr], *new_value);
            }
            Self::SwapInstructions { thread, instr_a, instr_b } => {
                mutant.threads[*thread].instructions.swap(*instr_a, *instr_b);
            }
            Self::DuplicateInstruction { thread, instr } => {
                let dup = mutant.threads[*thread].instructions[*instr].clone();
                mutant.threads[*thread].instructions.insert(*instr + 1, dup);
            }
            Self::RemoveInstruction { thread, instr } => {
                mutant.threads[*thread].instructions.remove(*instr);
            }
            Self::SwapThreads { thread_a, thread_b } => {
                mutant.threads.swap(*thread_a, *thread_b);
            }
            Self::AddThread { instructions } => {
                let tid = mutant.threads.len();
                mutant.threads.push(Thread::with_instructions(tid, instructions.clone()));
            }
            Self::RemoveThread { thread } => {
                mutant.threads.remove(*thread);
            }
            Self::LoadToRMW { thread, instr, write_value } => {
                if let Instruction::Load { reg, addr, ordering } =
                    mutant.threads[*thread].instructions[*instr]
                {
                    mutant.threads[*thread].instructions[*instr] =
                        Instruction::RMW { reg, addr, value: *write_value, ordering };
                }
            }
            Self::RMWToLoad { thread, instr } => {
                if let Instruction::RMW { reg, addr, ordering, .. } =
                    mutant.threads[*thread].instructions[*instr]
                {
                    mutant.threads[*thread].instructions[*instr] =
                        Instruction::Load { reg, addr, ordering };
                }
            }
            Self::StrengthenOrdering { thread, instr } => {
                let cur = get_ordering(&mutant.threads[*thread].instructions[*instr]);
                if let Some(stronger) = strengthen(cur) {
                    set_ordering(&mut mutant.threads[*thread].instructions[*instr], stronger);
                }
            }
            Self::WeakenOrdering { thread, instr } => {
                let cur = get_ordering(&mutant.threads[*thread].instructions[*instr]);
                if let Some(weaker) = weaken(cur) {
                    set_ordering(&mut mutant.threads[*thread].instructions[*instr], weaker);
                }
            }
            Self::ChangeFenceScope { thread, instr, new_scope } => {
                if let Instruction::Fence { scope, .. } =
                    &mut mutant.threads[*thread].instructions[*instr]
                {
                    *scope = *new_scope;
                }
            }
        }

        mutant.name = format!("{}-mut-{}", test.name, self);
        // Clear expected outcomes since mutation may invalidate them.
        mutant.expected_outcomes.clear();
        Some(mutant)
    }
}

// ---------------------------------------------------------------------------
// Ordering helpers
// ---------------------------------------------------------------------------

fn get_ordering(instr: &Instruction) -> Ordering {
    match instr {
        Instruction::Load { ordering, .. } => *ordering,
        Instruction::Store { ordering, .. } => *ordering,
        Instruction::Fence { ordering, .. } => *ordering,
        Instruction::RMW { ordering, .. } => *ordering,
        _ => Ordering::Relaxed,
    }
}

fn set_ordering(instr: &mut Instruction, ord: Ordering) {
    match instr {
        Instruction::Load { ordering, .. } => *ordering = ord,
        Instruction::Store { ordering, .. } => *ordering = ord,
        Instruction::Fence { ordering, .. } => *ordering = ord,
        Instruction::RMW { ordering, .. } => *ordering = ord,
        _ => {}
    }
}

fn set_address(instr: &mut Instruction, addr: Address) {
    match instr {
        Instruction::Load { addr: a, .. } => *a = addr,
        Instruction::Store { addr: a, .. } => *a = addr,
        Instruction::RMW { addr: a, .. } => *a = addr,
        _ => {}
    }
}

fn set_value(instr: &mut Instruction, val: Value) {
    match instr {
        Instruction::Store { value, .. } => *value = val,
        Instruction::RMW { value, .. } => *value = val,
        _ => {}
    }
}

/// Strengthen an ordering by one level.
fn strengthen(ord: Ordering) -> Option<Ordering> {
    match ord {
        Ordering::Relaxed => Some(Ordering::Acquire),
        Ordering::Acquire => Some(Ordering::SeqCst),
        Ordering::Release => Some(Ordering::SeqCst),
        Ordering::AcqRel => Some(Ordering::SeqCst),
        Ordering::AcquireCTA => Some(Ordering::AcquireGPU),
        Ordering::AcquireGPU => Some(Ordering::AcquireSystem),
        Ordering::ReleaseCTA => Some(Ordering::ReleaseGPU),
        Ordering::ReleaseGPU => Some(Ordering::ReleaseSystem),
        Ordering::SeqCst => None,
        Ordering::AcquireSystem => Some(Ordering::SeqCst),
        Ordering::ReleaseSystem => Some(Ordering::SeqCst),
    }
}

/// Weaken an ordering by one level.
fn weaken(ord: Ordering) -> Option<Ordering> {
    match ord {
        Ordering::SeqCst => Some(Ordering::AcqRel),
        Ordering::AcqRel => Some(Ordering::Acquire),
        Ordering::Acquire => Some(Ordering::Relaxed),
        Ordering::Release => Some(Ordering::Relaxed),
        Ordering::AcquireSystem => Some(Ordering::AcquireGPU),
        Ordering::AcquireGPU => Some(Ordering::AcquireCTA),
        Ordering::AcquireCTA => Some(Ordering::Relaxed),
        Ordering::ReleaseSystem => Some(Ordering::ReleaseGPU),
        Ordering::ReleaseGPU => Some(Ordering::ReleaseCTA),
        Ordering::ReleaseCTA => Some(Ordering::Relaxed),
        Ordering::Relaxed => None,
    }
}

// ---------------------------------------------------------------------------
// MutationResult
// ---------------------------------------------------------------------------

/// Result of applying a mutation.
#[derive(Debug, Clone)]
pub struct MutationResult {
    /// The original test.
    pub original: LitmusTest,
    /// The mutation applied.
    pub operator: MutationOperator,
    /// The mutated test (None if mutation was not applicable).
    pub mutant: Option<LitmusTest>,
    /// Whether the mutation changed the test's observable behavior.
    pub behavior_changed: Option<bool>,
}

impl MutationResult {
    /// Whether the mutation was successfully applied.
    pub fn success(&self) -> bool {
        self.mutant.is_some()
    }
}

// ---------------------------------------------------------------------------
// MutationConfig
// ---------------------------------------------------------------------------

/// Configuration for the mutation engine.
#[derive(Debug, Clone)]
pub struct MutationConfig {
    /// Orderings to use as replacement targets.
    pub target_orderings: Vec<Ordering>,
    /// Addresses to use as replacement targets.
    pub target_addresses: Vec<Address>,
    /// Values to use as replacement targets.
    pub target_values: Vec<Value>,
    /// Scopes to use for fence mutations.
    pub target_scopes: Vec<Scope>,
    /// Maximum number of mutations to apply in sequence.
    pub max_chain_length: usize,
    /// Whether to generate all single-point mutations.
    pub exhaustive: bool,
    /// Maximum number of mutants to produce.
    pub max_mutants: usize,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            target_orderings: vec![
                Ordering::Relaxed,
                Ordering::Acquire,
                Ordering::Release,
                Ordering::AcqRel,
                Ordering::SeqCst,
            ],
            target_addresses: vec![0x100, 0x108, 0x110],
            target_values: vec![0, 1, 2],
            target_scopes: vec![Scope::None, Scope::CTA, Scope::GPU, Scope::System],
            max_chain_length: 3,
            exhaustive: true,
            max_mutants: 10000,
        }
    }
}

impl MutationConfig {
    /// Config for GPU-aware mutations.
    pub fn gpu() -> Self {
        Self {
            target_orderings: vec![
                Ordering::Relaxed,
                Ordering::AcquireCTA, Ordering::ReleaseCTA,
                Ordering::AcquireGPU, Ordering::ReleaseGPU,
                Ordering::AcquireSystem, Ordering::ReleaseSystem,
                Ordering::SeqCst,
            ],
            target_scopes: vec![Scope::CTA, Scope::GPU, Scope::System],
            ..Default::default()
        }
    }

    /// Config for minimal mutations.
    pub fn minimal() -> Self {
        Self {
            target_orderings: vec![Ordering::Relaxed, Ordering::SeqCst],
            target_addresses: vec![0x100, 0x108],
            target_values: vec![1, 2],
            target_scopes: vec![Scope::None],
            max_chain_length: 1,
            max_mutants: 100,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// MutationEngine
// ---------------------------------------------------------------------------

/// Engine for systematically mutating litmus tests.
#[derive(Debug)]
pub struct MutationEngine {
    config: MutationConfig,
}

impl MutationEngine {
    /// Create a new mutation engine.
    pub fn new(config: MutationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_engine() -> Self {
        Self::new(MutationConfig::default())
    }

    /// Generate all single-point ordering mutations for a test.
    pub fn ordering_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                let cur_ord = get_ordering(instr);
                for &new_ord in &self.config.target_orderings {
                    if new_ord != cur_ord {
                        ops.push(MutationOperator::ChangeOrdering {
                            thread: tid,
                            instr: iid,
                            new_ordering: new_ord,
                        });
                    }
                }
            }
        }
        ops
    }

    /// Generate all address mutations.
    pub fn address_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                let cur_addr = get_address(instr);
                if let Some(cur) = cur_addr {
                    for &new_addr in &self.config.target_addresses {
                        if new_addr != cur {
                            ops.push(MutationOperator::ChangeAddress {
                                thread: tid,
                                instr: iid,
                                new_addr,
                            });
                        }
                    }
                }
            }
        }
        ops
    }

    /// Generate all value mutations.
    pub fn value_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                let cur_val = get_value(instr);
                if let Some(cur) = cur_val {
                    for &new_val in &self.config.target_values {
                        if new_val != cur {
                            ops.push(MutationOperator::ChangeValue {
                                thread: tid,
                                instr: iid,
                                new_value: new_val,
                            });
                        }
                    }
                }
            }
        }
        ops
    }

    /// Generate all fence insertion mutations.
    pub fn fence_insertion_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for pos in 0..=thread.instructions.len() {
                for &scope in &self.config.target_scopes {
                    ops.push(MutationOperator::InsertFence {
                        thread: tid,
                        position: pos,
                        ordering: Ordering::SeqCst,
                        scope,
                    });
                }
            }
        }
        ops
    }

    /// Generate all fence removal mutations.
    pub fn fence_removal_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                if matches!(instr, Instruction::Fence { .. }) {
                    ops.push(MutationOperator::RemoveFence {
                        thread: tid,
                        instr: iid,
                    });
                }
            }
        }
        ops
    }

    /// Generate instruction swap mutations.
    pub fn swap_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for i in 0..thread.instructions.len() {
                for j in i + 1..thread.instructions.len() {
                    ops.push(MutationOperator::SwapInstructions {
                        thread: tid,
                        instr_a: i,
                        instr_b: j,
                    });
                }
            }
        }
        ops
    }

    /// Generate thread swap mutations.
    pub fn thread_swap_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for i in 0..test.threads.len() {
            for j in i + 1..test.threads.len() {
                ops.push(MutationOperator::SwapThreads {
                    thread_a: i,
                    thread_b: j,
                });
            }
        }
        ops
    }

    /// Generate load-to-store and store-to-load mutations.
    pub fn load_store_swap_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                if matches!(instr, Instruction::Load { .. } | Instruction::Store { .. }) {
                    ops.push(MutationOperator::SwapLoadStore {
                        thread: tid,
                        instr: iid,
                    });
                }
            }
        }
        ops
    }

    /// Generate strengthen/weaken ordering mutations.
    pub fn strength_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                let ord = get_ordering(instr);
                if strengthen(ord).is_some() {
                    ops.push(MutationOperator::StrengthenOrdering {
                        thread: tid,
                        instr: iid,
                    });
                }
                if weaken(ord).is_some() {
                    ops.push(MutationOperator::WeakenOrdering {
                        thread: tid,
                        instr: iid,
                    });
                }
            }
        }
        ops
    }

    /// Generate load-to-RMW mutations.
    pub fn rmw_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                match instr {
                    Instruction::Load { .. } => {
                        for &val in &self.config.target_values {
                            if val > 0 {
                                ops.push(MutationOperator::LoadToRMW {
                                    thread: tid,
                                    instr: iid,
                                    write_value: val,
                                });
                            }
                        }
                    }
                    Instruction::RMW { .. } => {
                        ops.push(MutationOperator::RMWToLoad {
                            thread: tid,
                            instr: iid,
                        });
                    }
                    _ => {}
                }
            }
        }
        ops
    }

    /// Generate fence scope change mutations.
    pub fn scope_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut ops = Vec::new();
        for (tid, thread) in test.threads.iter().enumerate() {
            for (iid, instr) in thread.instructions.iter().enumerate() {
                if let Instruction::Fence { scope, .. } = instr {
                    for &new_scope in &self.config.target_scopes {
                        if new_scope != *scope {
                            ops.push(MutationOperator::ChangeFenceScope {
                                thread: tid,
                                instr: iid,
                                new_scope,
                            });
                        }
                    }
                }
            }
        }
        ops
    }

    /// Generate all single-point mutations.
    pub fn all_single_mutations(&self, test: &LitmusTest) -> Vec<MutationOperator> {
        let mut all = Vec::new();
        all.extend(self.ordering_mutations(test));
        all.extend(self.address_mutations(test));
        all.extend(self.value_mutations(test));
        all.extend(self.fence_insertion_mutations(test));
        all.extend(self.fence_removal_mutations(test));
        all.extend(self.swap_mutations(test));
        all.extend(self.thread_swap_mutations(test));
        all.extend(self.load_store_swap_mutations(test));
        all.extend(self.strength_mutations(test));
        all.extend(self.rmw_mutations(test));
        all.extend(self.scope_mutations(test));
        all
    }

    /// Apply all single-point mutations and return results.
    pub fn mutate_all(&self, test: &LitmusTest) -> Vec<MutationResult> {
        let ops = self.all_single_mutations(test);
        let mut results = Vec::new();

        for op in ops {
            if results.len() >= self.config.max_mutants { break; }
            let mutant = op.apply(test);
            results.push(MutationResult {
                original: test.clone(),
                operator: op,
                mutant,
                behavior_changed: None,
            });
        }
        results
    }

    /// Apply only ordering mutations and return results.
    pub fn mutate_orderings(&self, test: &LitmusTest) -> Vec<MutationResult> {
        let ops = self.ordering_mutations(test);
        ops.into_iter().map(|op| {
            let mutant = op.apply(test);
            MutationResult {
                original: test.clone(),
                operator: op,
                mutant,
                behavior_changed: None,
            }
        }).collect()
    }

    /// Apply mutation chains (sequences of mutations).
    pub fn mutate_chain(&self, test: &LitmusTest, chain_length: usize) -> Vec<MutationResult> {
        if chain_length == 0 { return vec![]; }

        let first_mutations = self.ordering_mutations(test);
        let mut results = Vec::new();

        for op in &first_mutations {
            if results.len() >= self.config.max_mutants { break; }
            if let Some(mutant) = op.apply(test) {
                if chain_length == 1 {
                    results.push(MutationResult {
                        original: test.clone(),
                        operator: op.clone(),
                        mutant: Some(mutant),
                        behavior_changed: None,
                    });
                } else {
                    // Apply further mutations to the mutant.
                    let further = self.ordering_mutations(&mutant);
                    for op2 in further.iter().take(5) {
                        if results.len() >= self.config.max_mutants { break; }
                        let mutant2 = op2.apply(&mutant);
                        results.push(MutationResult {
                            original: test.clone(),
                            operator: op2.clone(),
                            mutant: mutant2,
                            behavior_changed: None,
                        });
                    }
                }
            }
        }
        results
    }

    /// Count the number of possible single-point mutations.
    pub fn mutation_count(&self, test: &LitmusTest) -> usize {
        self.all_single_mutations(test).len()
    }

    /// Generate a mutation score summary.
    pub fn mutation_score(&self, test: &LitmusTest, killed: usize, total: usize) -> MutationScore {
        MutationScore {
            test_name: test.name.clone(),
            total_mutants: total,
            killed_mutants: killed,
            survived_mutants: total.saturating_sub(killed),
            score: if total > 0 { killed as f64 / total as f64 } else { 0.0 },
        }
    }
}

/// Mutation testing score.
#[derive(Debug, Clone)]
pub struct MutationScore {
    pub test_name: String,
    pub total_mutants: usize,
    pub killed_mutants: usize,
    pub survived_mutants: usize,
    pub score: f64,
}

impl fmt::Display for MutationScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "{}: {}/{} killed ({:.1}%)",
            self.test_name, self.killed_mutants, self.total_mutants,
            self.score * 100.0
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_address(instr: &Instruction) -> Option<Address> {
    match instr {
        Instruction::Load { addr, .. } => Some(*addr),
        Instruction::Store { addr, .. } => Some(*addr),
        Instruction::RMW { addr, .. } => Some(*addr),
        _ => None,
    }
}

fn get_value(instr: &Instruction) -> Option<Value> {
    match instr {
        Instruction::Store { value, .. } => Some(*value),
        Instruction::RMW { value, .. } => Some(*value),
        _ => None,
    }
}

/// Compute a diff between two tests showing what changed.
pub fn diff_tests(original: &LitmusTest, mutant: &LitmusTest) -> Vec<String> {
    let mut diffs = Vec::new();

    if original.threads.len() != mutant.threads.len() {
        diffs.push(format!(
            "Thread count: {} → {}", original.threads.len(), mutant.threads.len()
        ));
    }

    let min_threads = original.threads.len().min(mutant.threads.len());
    for tid in 0..min_threads {
        let t_orig = &original.threads[tid];
        let t_mut = &mutant.threads[tid];

        if t_orig.instructions.len() != t_mut.instructions.len() {
            diffs.push(format!(
                "T{} instruction count: {} → {}",
                tid, t_orig.instructions.len(), t_mut.instructions.len()
            ));
            continue;
        }

        for iid in 0..t_orig.instructions.len() {
            if t_orig.instructions[iid] != t_mut.instructions[iid] {
                diffs.push(format!(
                    "T{}[{}]: {} → {}",
                    tid, iid, t_orig.instructions[iid], t_mut.instructions[iid]
                ));
            }
        }
    }

    diffs
}

/// Classify a mutation by its type.
pub fn classify_mutation(op: &MutationOperator) -> &'static str {
    match op {
        MutationOperator::ChangeOrdering { .. } => "ordering",
        MutationOperator::SwapLoadStore { .. } => "load-store-swap",
        MutationOperator::InsertFence { .. } => "fence-insert",
        MutationOperator::RemoveFence { .. } => "fence-remove",
        MutationOperator::ChangeAddress { .. } => "address",
        MutationOperator::ChangeValue { .. } => "value",
        MutationOperator::SwapInstructions { .. } => "instruction-swap",
        MutationOperator::DuplicateInstruction { .. } => "duplicate",
        MutationOperator::RemoveInstruction { .. } => "remove",
        MutationOperator::SwapThreads { .. } => "thread-swap",
        MutationOperator::AddThread { .. } => "add-thread",
        MutationOperator::RemoveThread { .. } => "remove-thread",
        MutationOperator::LoadToRMW { .. } => "load-to-rmw",
        MutationOperator::RMWToLoad { .. } => "rmw-to-load",
        MutationOperator::StrengthenOrdering { .. } => "strengthen",
        MutationOperator::WeakenOrdering { .. } => "weaken",
        MutationOperator::ChangeFenceScope { .. } => "scope-change",
    }
}

/// Group mutations by category.
pub fn group_mutations(ops: &[MutationOperator]) -> HashMap<&'static str, Vec<&MutationOperator>> {
    let mut groups: HashMap<&'static str, Vec<&MutationOperator>> = HashMap::new();
    for op in ops {
        groups.entry(classify_mutation(op)).or_default().push(op);
    }
    groups
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_test() -> LitmusTest {
        crate::checker::litmus::mp_test()
    }

    fn make_sb_test() -> LitmusTest {
        crate::checker::litmus::sb_test()
    }

    fn make_test_with_fence() -> LitmusTest {
        let mut test = crate::checker::litmus::mp_test();
        test.threads[0].instructions.insert(1, Instruction::Fence {
            ordering: Ordering::SeqCst,
            scope: Scope::None,
        });
        test
    }

    #[test]
    fn test_mutation_config_default() {
        let config = MutationConfig::default();
        assert_eq!(config.max_chain_length, 3);
        assert!(config.exhaustive);
    }

    #[test]
    fn test_mutation_config_gpu() {
        let config = MutationConfig::gpu();
        assert!(config.target_orderings.contains(&Ordering::AcquireCTA));
        assert!(config.target_scopes.contains(&Scope::CTA));
    }

    #[test]
    fn test_mutation_config_minimal() {
        let config = MutationConfig::minimal();
        assert_eq!(config.target_orderings.len(), 2);
        assert_eq!(config.max_chain_length, 1);
    }

    #[test]
    fn test_change_ordering_applicable() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        };
        assert!(op.applicable(&test));
    }

    #[test]
    fn test_change_ordering_out_of_bounds() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeOrdering {
            thread: 10, instr: 0, new_ordering: Ordering::SeqCst,
        };
        assert!(!op.applicable(&test));
    }

    #[test]
    fn test_change_ordering_apply() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        };
        let mutant = op.apply(&test).unwrap();
        let ord = get_ordering(&mutant.threads[0].instructions[0]);
        assert_eq!(ord, Ordering::SeqCst);
    }

    #[test]
    fn test_swap_load_store() {
        let test = make_simple_test();
        let op = MutationOperator::SwapLoadStore { thread: 1, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        assert!(matches!(mutant.threads[1].instructions[0], Instruction::Store { .. }));
    }

    #[test]
    fn test_insert_fence() {
        let test = make_simple_test();
        let orig_len = test.threads[0].instructions.len();
        let op = MutationOperator::InsertFence {
            thread: 0, position: 1,
            ordering: Ordering::SeqCst, scope: Scope::None,
        };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions.len(), orig_len + 1);
        assert!(matches!(mutant.threads[0].instructions[1], Instruction::Fence { .. }));
    }

    #[test]
    fn test_remove_fence() {
        let test = make_test_with_fence();
        let orig_len = test.threads[0].instructions.len();
        let op = MutationOperator::RemoveFence { thread: 0, instr: 1 };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions.len(), orig_len - 1);
    }

    #[test]
    fn test_remove_fence_not_applicable() {
        let test = make_simple_test();
        let op = MutationOperator::RemoveFence { thread: 0, instr: 0 };
        assert!(!op.applicable(&test));
    }

    #[test]
    fn test_change_address() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeAddress {
            thread: 0, instr: 0, new_addr: 0x200,
        };
        let mutant = op.apply(&test).unwrap();
        let addr = get_address(&mutant.threads[0].instructions[0]).unwrap();
        assert_eq!(addr, 0x200);
    }

    #[test]
    fn test_change_value() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeValue {
            thread: 0, instr: 0, new_value: 42,
        };
        let mutant = op.apply(&test).unwrap();
        assert!(mutant.name.contains("mut"));
    }

    #[test]
    fn test_swap_instructions() {
        let test = make_simple_test();
        let op = MutationOperator::SwapInstructions {
            thread: 0, instr_a: 0, instr_b: 1,
        };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions[0], test.threads[0].instructions[1]);
        assert_eq!(mutant.threads[0].instructions[1], test.threads[0].instructions[0]);
    }

    #[test]
    fn test_duplicate_instruction() {
        let test = make_simple_test();
        let orig_len = test.threads[0].instructions.len();
        let op = MutationOperator::DuplicateInstruction { thread: 0, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions.len(), orig_len + 1);
    }

    #[test]
    fn test_remove_instruction() {
        let test = make_simple_test();
        let orig_len = test.threads[0].instructions.len();
        let op = MutationOperator::RemoveInstruction { thread: 0, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions.len(), orig_len - 1);
    }

    #[test]
    fn test_swap_threads() {
        let test = make_simple_test();
        let op = MutationOperator::SwapThreads { thread_a: 0, thread_b: 1 };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads[0].instructions, test.threads[1].instructions);
    }

    #[test]
    fn test_add_thread() {
        let test = make_simple_test();
        let op = MutationOperator::AddThread {
            instructions: vec![Instruction::Load { reg: 0, addr: 0x100, ordering: Ordering::Relaxed }],
        };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads.len(), test.threads.len() + 1);
    }

    #[test]
    fn test_remove_thread() {
        let test = make_simple_test();
        let op = MutationOperator::RemoveThread { thread: 0 };
        let mutant = op.apply(&test).unwrap();
        assert_eq!(mutant.threads.len(), test.threads.len() - 1);
    }

    #[test]
    fn test_remove_last_thread_not_allowed() {
        let mut test = LitmusTest::new("single");
        test.add_thread(Thread::new(0));
        let op = MutationOperator::RemoveThread { thread: 0 };
        assert!(!op.applicable(&test));
    }

    #[test]
    fn test_load_to_rmw() {
        let test = make_simple_test();
        // Thread 1, instr 0 is a load.
        let op = MutationOperator::LoadToRMW {
            thread: 1, instr: 0, write_value: 42,
        };
        let mutant = op.apply(&test).unwrap();
        assert!(matches!(mutant.threads[1].instructions[0], Instruction::RMW { .. }));
    }

    #[test]
    fn test_rmw_to_load() {
        let mut test = make_simple_test();
        test.threads[0].instructions[0] = Instruction::RMW {
            reg: 0, addr: 0x100, value: 1, ordering: Ordering::Relaxed,
        };
        let op = MutationOperator::RMWToLoad { thread: 0, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        assert!(matches!(mutant.threads[0].instructions[0], Instruction::Load { .. }));
    }

    #[test]
    fn test_strengthen_ordering() {
        let test = make_simple_test();
        let op = MutationOperator::StrengthenOrdering { thread: 0, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        let ord = get_ordering(&mutant.threads[0].instructions[0]);
        assert_ne!(ord, Ordering::Relaxed);
    }

    #[test]
    fn test_weaken_seq_cst() {
        let mut test = make_simple_test();
        set_ordering(&mut test.threads[0].instructions[0], Ordering::SeqCst);
        let op = MutationOperator::WeakenOrdering { thread: 0, instr: 0 };
        let mutant = op.apply(&test).unwrap();
        let ord = get_ordering(&mutant.threads[0].instructions[0]);
        assert_ne!(ord, Ordering::SeqCst);
    }

    #[test]
    fn test_change_fence_scope() {
        let test = make_test_with_fence();
        let op = MutationOperator::ChangeFenceScope {
            thread: 0, instr: 1, new_scope: Scope::CTA,
        };
        let mutant = op.apply(&test).unwrap();
        if let Instruction::Fence { scope, .. } = mutant.threads[0].instructions[1] {
            assert_eq!(scope, Scope::CTA);
        } else {
            panic!("Expected fence");
        }
    }

    #[test]
    fn test_mutation_clears_outcomes() {
        let test = make_simple_test();
        assert!(!test.expected_outcomes.is_empty());
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        };
        let mutant = op.apply(&test).unwrap();
        assert!(mutant.expected_outcomes.is_empty());
    }

    #[test]
    fn test_engine_ordering_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.ordering_mutations(&test);
        assert!(!ops.is_empty());
        for op in &ops {
            assert!(matches!(op, MutationOperator::ChangeOrdering { .. }));
        }
    }

    #[test]
    fn test_engine_address_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.address_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_value_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.value_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_fence_insertion() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.fence_insertion_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_fence_removal_none() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.fence_removal_mutations(&test);
        assert!(ops.is_empty());
    }

    #[test]
    fn test_engine_fence_removal_with_fence() {
        let test = make_test_with_fence();
        let engine = MutationEngine::default_engine();
        let ops = engine.fence_removal_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_swap_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.swap_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_thread_swap_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.thread_swap_mutations(&test);
        assert_eq!(ops.len(), 1); // 2 threads → 1 swap
    }

    #[test]
    fn test_engine_load_store_swap() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.load_store_swap_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_strength_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.strength_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_rmw_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.rmw_mutations(&test);
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_engine_all_single_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let all = engine.all_single_mutations(&test);
        assert!(!all.is_empty());
        // Should include multiple categories
        let groups = group_mutations(&all);
        assert!(groups.len() > 3);
    }

    #[test]
    fn test_engine_mutate_all() {
        let test = make_simple_test();
        let config = MutationConfig::minimal();
        let engine = MutationEngine::new(config);
        let results = engine.mutate_all(&test);
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.success());
        }
    }

    #[test]
    fn test_engine_mutate_orderings() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let results = engine.mutate_orderings(&test);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_mutate_chain() {
        let test = make_simple_test();
        let engine = MutationEngine::new(MutationConfig::minimal());
        let results = engine.mutate_chain(&test, 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_mutation_count() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let count = engine.mutation_count(&test);
        assert!(count > 0);
    }

    #[test]
    fn test_mutation_score() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let score = engine.mutation_score(&test, 8, 10);
        assert_eq!(score.total_mutants, 10);
        assert_eq!(score.killed_mutants, 8);
        assert_eq!(score.survived_mutants, 2);
        assert!((score.score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_mutation_score_display() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let score = engine.mutation_score(&test, 8, 10);
        let s = format!("{}", score);
        assert!(s.contains("80.0%"));
    }

    #[test]
    fn test_strengthen() {
        assert_eq!(strengthen(Ordering::Relaxed), Some(Ordering::Acquire));
        assert_eq!(strengthen(Ordering::Acquire), Some(Ordering::SeqCst));
        assert_eq!(strengthen(Ordering::SeqCst), None);
    }

    #[test]
    fn test_weaken() {
        assert_eq!(weaken(Ordering::SeqCst), Some(Ordering::AcqRel));
        assert_eq!(weaken(Ordering::Relaxed), None);
    }

    #[test]
    fn test_diff_tests_same() {
        let test = make_simple_test();
        let diffs = diff_tests(&test, &test);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_diff_tests_different() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        };
        let mutant = op.apply(&test).unwrap();
        let diffs = diff_tests(&test, &mutant);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_diff_tests_different_thread_count() {
        let test = make_simple_test();
        let op = MutationOperator::RemoveThread { thread: 0 };
        let mutant = op.apply(&test).unwrap();
        let diffs = diff_tests(&test, &mutant);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_classify_mutation() {
        assert_eq!(classify_mutation(&MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        }), "ordering");
        assert_eq!(classify_mutation(&MutationOperator::InsertFence {
            thread: 0, position: 0, ordering: Ordering::SeqCst, scope: Scope::None,
        }), "fence-insert");
    }

    #[test]
    fn test_group_mutations() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let ops = engine.all_single_mutations(&test);
        let groups = group_mutations(&ops);
        assert!(groups.contains_key("ordering"));
    }

    #[test]
    fn test_get_address_load() {
        let instr = Instruction::Load { reg: 0, addr: 0x100, ordering: Ordering::Relaxed };
        assert_eq!(get_address(&instr), Some(0x100));
    }

    #[test]
    fn test_get_address_fence() {
        let instr = Instruction::Fence { ordering: Ordering::SeqCst, scope: Scope::None };
        assert_eq!(get_address(&instr), None);
    }

    #[test]
    fn test_get_value_store() {
        let instr = Instruction::Store { addr: 0x100, value: 42, ordering: Ordering::Relaxed };
        assert_eq!(get_value(&instr), Some(42));
    }

    #[test]
    fn test_get_value_load() {
        let instr = Instruction::Load { reg: 0, addr: 0x100, ordering: Ordering::Relaxed };
        assert_eq!(get_value(&instr), None);
    }

    #[test]
    fn test_mutation_operator_display() {
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 1, new_ordering: Ordering::SeqCst,
        };
        let s = format!("{}", op);
        assert!(s.contains("ChangeOrdering"));
    }

    #[test]
    fn test_mutation_result_success() {
        let test = make_simple_test();
        let op = MutationOperator::ChangeOrdering {
            thread: 0, instr: 0, new_ordering: Ordering::SeqCst,
        };
        let mutant = op.apply(&test);
        let result = MutationResult {
            original: test,
            operator: op,
            mutant,
            behavior_changed: None,
        };
        assert!(result.success());
    }

    #[test]
    fn test_scope_mutations() {
        let test = make_test_with_fence();
        let engine = MutationEngine::default_engine();
        let ops = engine.scope_mutations(&test);
        // The fence has Scope::None, so there should be mutations to CTA, GPU, System
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_sb_mutations() {
        let test = make_sb_test();
        let engine = MutationEngine::new(MutationConfig::minimal());
        let results = engine.mutate_all(&test);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_mutation_chain_zero_length() {
        let test = make_simple_test();
        let engine = MutationEngine::default_engine();
        let results = engine.mutate_chain(&test, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_strengthen_gpu_orderings() {
        assert_eq!(strengthen(Ordering::AcquireCTA), Some(Ordering::AcquireGPU));
        assert_eq!(strengthen(Ordering::AcquireGPU), Some(Ordering::AcquireSystem));
        assert_eq!(strengthen(Ordering::ReleaseCTA), Some(Ordering::ReleaseGPU));
        assert_eq!(strengthen(Ordering::ReleaseGPU), Some(Ordering::ReleaseSystem));
    }

    #[test]
    fn test_weaken_gpu_orderings() {
        assert_eq!(weaken(Ordering::AcquireSystem), Some(Ordering::AcquireGPU));
        assert_eq!(weaken(Ordering::AcquireGPU), Some(Ordering::AcquireCTA));
        assert_eq!(weaken(Ordering::AcquireCTA), Some(Ordering::Relaxed));
        assert_eq!(weaken(Ordering::ReleaseSystem), Some(Ordering::ReleaseGPU));
        assert_eq!(weaken(Ordering::ReleaseGPU), Some(Ordering::ReleaseCTA));
        assert_eq!(weaken(Ordering::ReleaseCTA), Some(Ordering::Relaxed));
    }
}
