//! Litmus test generator.
//!
//! Provides systematic litmus test generation, random test generation,
//! interesting test heuristics, test minimization, and test mutation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome,
};
use crate::checker::execution::Address;

// ---------------------------------------------------------------------------
// TestGenerator — systematic test generation
// ---------------------------------------------------------------------------

/// Systematic litmus test generator.
pub struct TestGenerator {
    /// Maximum number of threads.
    pub max_threads: usize,
    /// Maximum instructions per thread.
    pub max_instructions: usize,
    /// Addresses to use.
    pub addresses: Vec<Address>,
    /// Values to write.
    pub values: Vec<u64>,
    /// Orderings to use.
    pub orderings: Vec<Ordering>,
}

impl TestGenerator {
    /// Create a default generator.
    pub fn new() -> Self {
        Self {
            max_threads: 2,
            max_instructions: 2,
            addresses: vec![0x100, 0x200],
            values: vec![1],
            orderings: vec![Ordering::Relaxed],
        }
    }

    /// Set max threads.
    pub fn with_max_threads(mut self, n: usize) -> Self {
        self.max_threads = n;
        self
    }

    /// Set max instructions per thread.
    pub fn with_max_instructions(mut self, n: usize) -> Self {
        self.max_instructions = n;
        self
    }

    /// Set addresses.
    pub fn with_addresses(mut self, addrs: Vec<Address>) -> Self {
        self.addresses = addrs;
        self
    }

    /// Set orderings.
    pub fn with_orderings(mut self, ords: Vec<Ordering>) -> Self {
        self.orderings = ords;
        self
    }

    /// Generate all tests up to the given bounds.
    pub fn generate_all(&self) -> Vec<LitmusTest> {
        let mut tests = Vec::new();

        // Generate for each thread count.
        for n_threads in 2..=self.max_threads {
            for n_instrs in 1..=self.max_instructions {
                let thread_configs = self.generate_thread_configs(n_instrs);
                let combos = self.thread_combinations(&thread_configs, n_threads);
                for (idx, combo) in combos.iter().enumerate() {
                    let mut test = LitmusTest::new(
                        &format!("gen_{}t_{}i_{}", n_threads, n_instrs, idx)
                    );
                    for addr in &self.addresses {
                        test.set_initial(*addr, 0);
                    }
                    for (tid, thread_instrs) in combo.iter().enumerate() {
                        test.add_thread(Thread::with_instructions(tid, thread_instrs.clone()));
                    }
                    tests.push(test);
                }
            }
        }

        tests
    }

    /// Generate single-thread instruction sequences.
    fn generate_thread_configs(&self, max_instrs: usize) -> Vec<Vec<Instruction>> {
        let mut configs = Vec::new();
        let ops: Vec<(&str, Address)> = self.addresses.iter()
            .flat_map(|&addr| {
                vec![("load", addr), ("store", addr)]
            })
            .collect();

        // Generate all combinations up to max_instrs.
        for len in 1..=max_instrs {
            self.gen_instr_combos(&ops, len, &mut vec![], &mut configs);
        }

        // Limit to avoid explosion.
        configs.truncate(100);
        configs
    }

    fn gen_instr_combos(
        &self,
        ops: &[(&str, Address)],
        remaining: usize,
        current: &mut Vec<Instruction>,
        results: &mut Vec<Vec<Instruction>>,
    ) {
        if remaining == 0 {
            results.push(current.clone());
            return;
        }
        if results.len() >= 100 { return; }

        for &(op, addr) in ops {
            for &ordering in &self.orderings {
                let instr = match op {
                    "load" => Instruction::Load {
                        reg: current.len(),
                        addr,
                        ordering,
                    },
                    "store" => Instruction::Store {
                        addr,
                        value: self.values[0],
                        ordering,
                    },
                    _ => continue,
                };
                current.push(instr);
                self.gen_instr_combos(ops, remaining - 1, current, results);
                current.pop();
            }
        }
    }

    fn thread_combinations(
        &self,
        configs: &[Vec<Instruction>],
        n_threads: usize,
    ) -> Vec<Vec<Vec<Instruction>>> {
        let mut result = Vec::new();
        self.thread_combo_helper(configs, n_threads, &mut vec![], &mut result);
        // Limit output.
        result.truncate(50);
        result
    }

    fn thread_combo_helper(
        &self,
        configs: &[Vec<Instruction>],
        remaining: usize,
        current: &mut Vec<Vec<Instruction>>,
        results: &mut Vec<Vec<Vec<Instruction>>>,
    ) {
        if remaining == 0 {
            results.push(current.clone());
            return;
        }
        if results.len() >= 50 { return; }

        for config in configs {
            current.push(config.clone());
            self.thread_combo_helper(configs, remaining - 1, current, results);
            current.pop();
        }
    }
}

impl Default for TestGenerator {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Interesting test heuristics
// ---------------------------------------------------------------------------

/// Filter tests to only include "interesting" ones (those that may exhibit
/// relaxed behaviors).
pub fn filter_interesting(tests: &[LitmusTest]) -> Vec<LitmusTest> {
    tests.iter()
        .filter(|t| is_interesting(t))
        .cloned()
        .collect()
}

/// Check if a test is "interesting" (may exhibit relaxed behaviors).
fn is_interesting(test: &LitmusTest) -> bool {
    // A test is interesting if:
    // 1. It has multiple threads accessing shared addresses.
    // 2. At least one thread writes and another reads the same address.
    let all_addrs = test.all_addresses();
    if all_addrs.is_empty() { return false; }

    for &addr in &all_addrs {
        let mut has_writer = false;
        let mut has_reader = false;
        let mut writer_thread = None;

        for thread in &test.threads {
            for instr in &thread.instructions {
                match instr {
                    Instruction::Store { addr: a, .. } if *a == addr => {
                        has_writer = true;
                        writer_thread = Some(thread.id);
                    }
                    Instruction::Load { addr: a, .. } if *a == addr => {
                        has_reader = true;
                        if writer_thread.is_some() && writer_thread != Some(thread.id) {
                            return true; // Cross-thread communication.
                        }
                    }
                    _ => {}
                }
            }
        }
        if has_writer && has_reader { return true; }
    }
    false
}

// ---------------------------------------------------------------------------
// Random test generation
// ---------------------------------------------------------------------------

/// Generate random litmus tests.
pub struct RandomTestGenerator {
    pub n_threads: usize,
    pub n_instructions: usize,
    pub addresses: Vec<Address>,
    pub values: Vec<u64>,
}

impl RandomTestGenerator {
    pub fn new(n_threads: usize, n_instructions: usize) -> Self {
        Self {
            n_threads,
            n_instructions,
            addresses: vec![0x100, 0x200],
            values: vec![1, 2],
        }
    }

    /// Generate a single random test.
    pub fn generate_one(&self) -> LitmusTest {
        let mut rng = rand::thread_rng();
        let mut test = LitmusTest::new(&format!("random_{}", rng.gen::<u32>()));

        for &addr in &self.addresses {
            test.set_initial(addr, 0);
        }

        for tid in 0..self.n_threads {
            let mut thread = Thread::new(tid);
            for i in 0..self.n_instructions {
                let addr = self.addresses[rng.gen_range(0..self.addresses.len())];
                if rng.gen_bool(0.5) {
                    thread.load(i, addr, Ordering::Relaxed);
                } else {
                    let val = self.values[rng.gen_range(0..self.values.len())];
                    thread.store(addr, val, Ordering::Relaxed);
                }
            }
            test.add_thread(thread);
        }

        test
    }

    /// Generate N random tests.
    pub fn generate(&self, n: usize) -> Vec<LitmusTest> {
        (0..n).map(|_| self.generate_one()).collect()
    }
}

// ---------------------------------------------------------------------------
// Test minimization
// ---------------------------------------------------------------------------

/// Minimize a litmus test by removing redundant instructions.
/// An instruction is redundant if removing it doesn't change the test's
/// observable behavior (with respect to the expected outcomes).
pub fn minimize_test(test: &LitmusTest) -> LitmusTest {
    let mut minimized = test.clone();

    // Try removing each instruction from each thread.
    for tid in 0..minimized.threads.len() {
        let mut i = 0;
        while i < minimized.threads[tid].instructions.len() {
            let instr = minimized.threads[tid].instructions[i].clone();

            // Don't remove the only instruction.
            if minimized.threads[tid].instructions.len() <= 1 {
                i += 1;
                continue;
            }

            minimized.threads[tid].instructions.remove(i);

            // Check if the test is still interesting.
            if !is_interesting(&minimized) {
                // Put it back.
                minimized.threads[tid].instructions.insert(i, instr);
                i += 1;
            }
            // else: successfully removed.
        }
    }

    // Remove empty threads.
    minimized.threads.retain(|t| !t.instructions.is_empty());

    minimized
}

// ---------------------------------------------------------------------------
// Test mutation
// ---------------------------------------------------------------------------

/// Mutation operations for litmus tests.
pub enum Mutation {
    /// Flip the ordering of an instruction.
    FlipOrdering {
        thread: usize,
        instruction: usize,
        new_ordering: Ordering,
    },
    /// Add a fence before an instruction.
    AddFence {
        thread: usize,
        position: usize,
        ordering: Ordering,
    },
    /// Remove an instruction.
    RemoveInstruction {
        thread: usize,
        instruction: usize,
    },
    /// Swap two instructions within a thread.
    SwapInstructions {
        thread: usize,
        i: usize,
        j: usize,
    },
}

/// Apply a mutation to a litmus test.
pub fn apply_mutation(test: &LitmusTest, mutation: &Mutation) -> LitmusTest {
    let mut mutated = test.clone();
    match mutation {
        Mutation::FlipOrdering { thread, instruction, new_ordering } => {
            if *thread < mutated.threads.len() &&
               *instruction < mutated.threads[*thread].instructions.len() {
                let instr = &mut mutated.threads[*thread].instructions[*instruction];
                match instr {
                    Instruction::Load { ordering, .. } => *ordering = *new_ordering,
                    Instruction::Store { ordering, .. } => *ordering = *new_ordering,
                    Instruction::RMW { ordering, .. } => *ordering = *new_ordering,
                    _ => {}
                }
            }
        }
        Mutation::AddFence { thread, position, ordering } => {
            if *thread < mutated.threads.len() {
                let fence = Instruction::Fence {
                    ordering: *ordering,
                    scope: crate::checker::litmus::Scope::None,
                };
                let pos = (*position).min(mutated.threads[*thread].instructions.len());
                mutated.threads[*thread].instructions.insert(pos, fence);
            }
        }
        Mutation::RemoveInstruction { thread, instruction } => {
            if *thread < mutated.threads.len() &&
               *instruction < mutated.threads[*thread].instructions.len() {
                mutated.threads[*thread].instructions.remove(*instruction);
            }
        }
        Mutation::SwapInstructions { thread, i, j } => {
            if *thread < mutated.threads.len() {
                let len = mutated.threads[*thread].instructions.len();
                if *i < len && *j < len {
                    mutated.threads[*thread].instructions.swap(*i, *j);
                }
            }
        }
    }
    mutated
}

/// Generate all single-mutation variants of a test.
pub fn generate_mutations(test: &LitmusTest) -> Vec<LitmusTest> {
    let mut variants = Vec::new();
    let orderings = [Ordering::Relaxed, Ordering::Acquire, Ordering::Release, Ordering::SeqCst];

    for (tid, thread) in test.threads.iter().enumerate() {
        for (iid, _instr) in thread.instructions.iter().enumerate() {
            // Flip orderings.
            for &ord in &orderings {
                let mutation = Mutation::FlipOrdering {
                    thread: tid,
                    instruction: iid,
                    new_ordering: ord,
                };
                variants.push(apply_mutation(test, &mutation));
            }

            // Add fences.
            for &ord in &orderings {
                let mutation = Mutation::AddFence {
                    thread: tid,
                    position: iid,
                    ordering: ord,
                };
                variants.push(apply_mutation(test, &mutation));
            }
        }

        // Swap pairs.
        for i in 0..thread.instructions.len() {
            for j in i + 1..thread.instructions.len() {
                let mutation = Mutation::SwapInstructions {
                    thread: tid, i, j,
                };
                variants.push(apply_mutation(test, &mutation));
            }
        }
    }

    variants
}

// ---------------------------------------------------------------------------
// Standard test library builder
// ---------------------------------------------------------------------------

/// Build standard litmus tests.
pub struct StandardTests;

impl StandardTests {
    /// Store Buffer (SB) test.
    pub fn sb() -> LitmusTest {
        let mut test = LitmusTest::new("SB");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t1);

        test.expect(
            Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
            LitmusOutcome::Forbidden,
        );
        test
    }

    /// Message Passing (MP) test.
    pub fn mp() -> LitmusTest {
        let mut test = LitmusTest::new("MP");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.store(0x200, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0x200, Ordering::Relaxed);
        t1.load(1, 0x100, Ordering::Relaxed);
        test.add_thread(t1);

        test.expect(
            Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
            LitmusOutcome::Forbidden,
        );
        test
    }

    /// Load Buffering (LB) test.
    pub fn lb() -> LitmusTest {
        let mut test = LitmusTest::new("LB");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.load(0, 0x100, Ordering::Relaxed);
        t0.store(0x200, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0x200, Ordering::Relaxed);
        t1.store(0x100, 1, Ordering::Relaxed);
        test.add_thread(t1);

        test.expect(
            Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
            LitmusOutcome::Forbidden,
        );
        test
    }

    /// 2+2W (two writes on each thread) test.
    pub fn two_plus_two_w() -> LitmusTest {
        let mut test = LitmusTest::new("2+2W");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.store(0x200, 2, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.store(0x100, 2, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_default() {
        let gen = TestGenerator::new();
        assert_eq!(gen.max_threads, 2);
        assert_eq!(gen.max_instructions, 2);
    }

    #[test]
    fn test_generator_with_config() {
        let gen = TestGenerator::new()
            .with_max_threads(3)
            .with_max_instructions(3);
        assert_eq!(gen.max_threads, 3);
        assert_eq!(gen.max_instructions, 3);
    }

    #[test]
    fn test_generate_all() {
        let gen = TestGenerator::new()
            .with_max_threads(2)
            .with_max_instructions(1);
        let tests = gen.generate_all();
        assert!(!tests.is_empty());
        for test in &tests {
            assert!(test.thread_count() >= 2);
        }
    }

    #[test]
    fn test_filter_interesting() {
        let gen = TestGenerator::new()
            .with_max_threads(2)
            .with_max_instructions(1);
        let tests = gen.generate_all();
        let interesting = filter_interesting(&tests);
        // All interesting tests should have cross-thread communication.
        for test in &interesting {
            assert!(is_interesting(test));
        }
    }

    #[test]
    fn test_random_generator() {
        let gen = RandomTestGenerator::new(2, 2);
        let test = gen.generate_one();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_random_generate_multiple() {
        let gen = RandomTestGenerator::new(2, 2);
        let tests = gen.generate(5);
        assert_eq!(tests.len(), 5);
    }

    #[test]
    fn test_sb_standard() {
        let sb = StandardTests::sb();
        assert_eq!(sb.name, "SB");
        assert_eq!(sb.thread_count(), 2);
        assert!(!sb.expected_outcomes.is_empty());
    }

    #[test]
    fn test_mp_standard() {
        let mp = StandardTests::mp();
        assert_eq!(mp.name, "MP");
        assert_eq!(mp.thread_count(), 2);
    }

    #[test]
    fn test_lb_standard() {
        let lb = StandardTests::lb();
        assert_eq!(lb.name, "LB");
        assert_eq!(lb.thread_count(), 2);
    }

    #[test]
    fn test_two_plus_two_w() {
        let t = StandardTests::two_plus_two_w();
        assert_eq!(t.name, "2+2W");
    }

    #[test]
    fn test_minimize() {
        let sb = StandardTests::sb();
        let minimized = minimize_test(&sb);
        // Minimized should still be interesting.
        if minimized.thread_count() >= 2 {
            assert!(is_interesting(&minimized));
        }
    }

    #[test]
    fn test_flip_ordering_mutation() {
        let sb = StandardTests::sb();
        let mutation = Mutation::FlipOrdering {
            thread: 0,
            instruction: 0,
            new_ordering: Ordering::SeqCst,
        };
        let mutated = apply_mutation(&sb, &mutation);
        assert_eq!(mutated.thread_count(), 2);
    }

    #[test]
    fn test_add_fence_mutation() {
        let sb = StandardTests::sb();
        let mutation = Mutation::AddFence {
            thread: 0,
            position: 1,
            ordering: Ordering::SeqCst,
        };
        let mutated = apply_mutation(&sb, &mutation);
        // Should have one more instruction.
        assert_eq!(
            mutated.threads[0].instructions.len(),
            sb.threads[0].instructions.len() + 1
        );
    }

    #[test]
    fn test_swap_mutation() {
        let sb = StandardTests::sb();
        let mutation = Mutation::SwapInstructions {
            thread: 0,
            i: 0,
            j: 1,
        };
        let mutated = apply_mutation(&sb, &mutation);
        assert_eq!(mutated.thread_count(), 2);
    }

    #[test]
    fn test_generate_mutations() {
        let sb = StandardTests::sb();
        let mutations = generate_mutations(&sb);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_is_interesting_sb() {
        let sb = StandardTests::sb();
        assert!(is_interesting(&sb));
    }

    #[test]
    fn test_is_not_interesting_empty() {
        let test = LitmusTest::new("empty");
        assert!(!is_interesting(&test));
    }
}
