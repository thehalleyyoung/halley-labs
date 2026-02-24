//! Scaled litmus test generator — 4-8 threads, 3-6 memory locations.
//!
//! Produces thousands to millions of execution graphs from:
//!   - Store buffering variants (N-thread generalization)
//!   - Message passing chains (N-way)
//!   - Read-modify-write patterns (CAS chains, fetch-add trees)
//!   - Dekker's, Peterson's, Lamport's bakery with real thread counts
//!   - IRIW generalizations (N readers, M writers)
//!   - Write-read causality (WRC) chains
//!   - Coherence (CoRR, CoWR, CoRW, CoWW) with multiple addresses

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, Scope, LitmusOutcome,
};
use crate::checker::execution::{Address, Value};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which dimension to scale along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalingDimension {
    /// Increase thread count (4..8).
    Threads,
    /// Increase memory locations (3..6).
    Locations,
    /// Increase instructions per thread.
    InstructionsPerThread,
    /// Combinatorial: threads × locations.
    Combined,
}

/// Family of litmus test patterns to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternFamily {
    StoreBuffering,
    MessagePassing,
    LoadBuffering,
    IndependentReadsIndependentWrites,
    WriteReadCausality,
    Coherence,
    ReadModifyWrite,
    Dekker,
    Peterson,
    LamportBakery,
    TreiberStack,
    MichaelScottQueue,
    TicketLock,
    MCSLock,
    SeqLock,
}

/// Configuration for scaled test generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaledTestConfig {
    /// Minimum thread count.
    pub min_threads: usize,
    /// Maximum thread count.
    pub max_threads: usize,
    /// Minimum number of memory locations.
    pub min_locations: usize,
    /// Maximum number of memory locations.
    pub max_locations: usize,
    /// Maximum instructions per thread.
    pub max_instrs_per_thread: usize,
    /// Which pattern families to generate.
    pub families: Vec<PatternFamily>,
    /// Memory orderings to explore.
    pub orderings: Vec<Ordering>,
    /// Whether to include scoped (GPU) variants.
    pub include_scoped: bool,
    /// Maximum number of tests to generate (0 = unlimited).
    pub max_tests: usize,
}

impl Default for ScaledTestConfig {
    fn default() -> Self {
        Self {
            min_threads: 4,
            max_threads: 8,
            min_locations: 3,
            max_locations: 6,
            max_instrs_per_thread: 8,
            families: vec![
                PatternFamily::StoreBuffering,
                PatternFamily::MessagePassing,
                PatternFamily::LoadBuffering,
                PatternFamily::IndependentReadsIndependentWrites,
                PatternFamily::WriteReadCausality,
                PatternFamily::Coherence,
                PatternFamily::ReadModifyWrite,
                PatternFamily::Dekker,
                PatternFamily::Peterson,
                PatternFamily::LamportBakery,
            ],
            orderings: vec![
                Ordering::Relaxed,
                Ordering::Acquire,
                Ordering::Release,
                Ordering::AcqRel,
                Ordering::SeqCst,
            ],
            include_scoped: true,
            max_tests: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Statistics for a batch of generated tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaledTestResult {
    pub pattern: PatternFamily,
    pub num_threads: usize,
    pub num_locations: usize,
    pub num_tests_generated: usize,
    pub estimated_execution_graphs: u64,
    pub generation_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Main generator for scaled litmus tests.
pub struct ScaledLitmusGenerator {
    config: ScaledTestConfig,
    generated: Vec<LitmusTest>,
    results: Vec<ScaledTestResult>,
}

impl ScaledLitmusGenerator {
    pub fn new(config: ScaledTestConfig) -> Self {
        Self {
            config,
            generated: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ScaledTestConfig::default())
    }

    /// Generate all configured test families across all scaling dimensions.
    pub fn generate_all(&mut self) -> &[LitmusTest] {
        let families = self.config.families.clone();
        for family in &families {
            for n_threads in self.config.min_threads..=self.config.max_threads {
                for n_locs in self.config.min_locations..=self.config.max_locations {
                    if self.config.max_tests > 0 && self.generated.len() >= self.config.max_tests {
                        return &self.generated;
                    }
                    let start = std::time::Instant::now();
                    let tests = match family {
                        PatternFamily::StoreBuffering =>
                            self.gen_store_buffering(n_threads, n_locs),
                        PatternFamily::MessagePassing =>
                            self.gen_message_passing(n_threads, n_locs),
                        PatternFamily::LoadBuffering =>
                            self.gen_load_buffering(n_threads, n_locs),
                        PatternFamily::IndependentReadsIndependentWrites =>
                            self.gen_iriw(n_threads, n_locs),
                        PatternFamily::WriteReadCausality =>
                            self.gen_wrc(n_threads, n_locs),
                        PatternFamily::Coherence =>
                            self.gen_coherence(n_threads, n_locs),
                        PatternFamily::ReadModifyWrite =>
                            self.gen_rmw_patterns(n_threads, n_locs),
                        PatternFamily::Dekker =>
                            self.gen_dekker(n_threads, n_locs),
                        PatternFamily::Peterson =>
                            self.gen_peterson(n_threads, n_locs),
                        PatternFamily::LamportBakery =>
                            self.gen_lamport_bakery(n_threads, n_locs),
                        PatternFamily::TreiberStack =>
                            self.gen_treiber_stack(n_threads, n_locs),
                        PatternFamily::MichaelScottQueue =>
                            self.gen_michael_scott_queue(n_threads, n_locs),
                        PatternFamily::TicketLock =>
                            self.gen_ticket_lock(n_threads, n_locs),
                        PatternFamily::MCSLock =>
                            self.gen_mcs_lock(n_threads, n_locs),
                        PatternFamily::SeqLock =>
                            self.gen_seqlock(n_threads, n_locs),
                    };
                    let elapsed = start.elapsed().as_millis() as u64;
                    let n_tests = tests.len();
                    let est_graphs = Self::estimate_execution_graphs(n_threads, n_locs, &tests);
                    self.results.push(ScaledTestResult {
                        pattern: *family,
                        num_threads: n_threads,
                        num_locations: n_locs,
                        num_tests_generated: n_tests,
                        estimated_execution_graphs: est_graphs,
                        generation_time_ms: elapsed,
                    });
                    self.generated.extend(tests);
                }
            }
        }
        &self.generated
    }

    /// Get generation statistics.
    pub fn results(&self) -> &[ScaledTestResult] {
        &self.results
    }

    /// Get all generated tests.
    pub fn tests(&self) -> &[LitmusTest] {
        &self.generated
    }

    // -----------------------------------------------------------------------
    // Estimation
    // -----------------------------------------------------------------------

    /// Estimate number of execution graphs for a set of tests.
    /// Uses the formula: |rf| × |co| ≈ (n_writes)^(n_reads) × (n_writes_per_loc!)^n_locs
    fn estimate_execution_graphs(
        n_threads: usize,
        n_locs: usize,
        tests: &[LitmusTest],
    ) -> u64 {
        if tests.is_empty() {
            return 0;
        }
        // Average across tests
        let mut total: u64 = 0;
        for test in tests {
            let n_reads: u64 = test.threads.iter()
                .map(|t| t.instructions.iter()
                    .filter(|i| matches!(i, Instruction::Load { .. }))
                    .count() as u64)
                .sum();
            let n_writes: u64 = test.threads.iter()
                .map(|t| t.instructions.iter()
                    .filter(|i| matches!(i, Instruction::Store { .. }))
                    .count() as u64)
                .sum();
            // rf choices: each read can read from any write to same address + init
            let rf = if n_writes > 0 { (n_writes + 1).saturating_pow(n_reads as u32) } else { 1 };
            // co choices: factorial of writes per location (simplified)
            let writes_per_loc = if n_locs > 0 { n_writes / n_locs as u64 } else { 0 };
            let co = factorial(writes_per_loc).saturating_pow(n_locs as u32);
            total = total.saturating_add(rf.saturating_mul(co));
        }
        total
    }

    // -----------------------------------------------------------------------
    // N-thread Store Buffering (SB)
    // -----------------------------------------------------------------------
    // Classic SB: Thread i writes x_i then reads x_{(i+1) % N}
    // All relaxed outcomes are the "surprise" — all reads see 0.

    fn gen_store_buffering(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let locs = n_locs.min(n_threads);

        // Generate for each ordering combination
        for &ord in &self.config.orderings {
            let name = format!("SB-{}-{}T-{}L-{}", "scaled", n_threads, locs, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads {
                let mut thread = Thread::new(t);
                let write_loc = (t % locs) as Address;
                let read_loc = ((t + 1) % locs) as Address;
                thread.store(write_loc, 1, ord);
                thread.load(t, read_loc, ord);
                test.add_thread(thread);
            }

            // Forbidden outcome: all reads see 0
            let mut outcome = Outcome::new();
            for t in 0..n_threads {
                outcome = outcome.with_reg(t, t, 0);
            }
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        // Scoped GPU variants
        if self.config.include_scoped {
            let scoped_orderings = vec![
                (Ordering::ReleaseCTA, Ordering::AcquireCTA, "cta"),
                (Ordering::ReleaseGPU, Ordering::AcquireGPU, "gpu"),
                (Ordering::ReleaseSystem, Ordering::AcquireSystem, "sys"),
            ];
            for (rel, acq, scope_name) in &scoped_orderings {
                let name = format!("SB-{}-{}T-{}L-{}", "scoped", n_threads, locs, scope_name);
                let mut test = LitmusTest::new(&name);
                for t in 0..n_threads {
                    let mut thread = Thread::new(t);
                    let write_loc = (t % locs) as Address;
                    let read_loc = ((t + 1) % locs) as Address;
                    thread.store(write_loc, 1, *rel);
                    thread.load(t, read_loc, *acq);
                    test.add_thread(thread);
                }
                let mut outcome = Outcome::new();
                for t in 0..n_threads {
                    outcome = outcome.with_reg(t, t, 0);
                }
                test.expect(outcome, LitmusOutcome::Forbidden);
                tests.push(test);
            }
        }

        tests
    }

    // -----------------------------------------------------------------------
    // N-thread Message Passing (MP)
    // -----------------------------------------------------------------------
    // Chain: T0 writes data+flag, T1 reads flag+data, T1 writes flag2,
    // T2 reads flag2+data, etc.

    fn gen_message_passing(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n_data_locs = n_locs.saturating_sub(1).max(1);

        for &ord in &self.config.orderings {
            let name = format!("MP-chain-{}T-{}L-{}", n_threads, n_locs, ord);
            let mut test = LitmusTest::new(&name);

            // Thread 0: writes data locations then sets flag
            let mut t0 = Thread::new(0);
            for loc in 0..n_data_locs {
                t0.store(loc as Address, 1, ord);
            }
            t0.store(n_data_locs as Address, 1, ord); // flag
            test.add_thread(t0);

            // Intermediate threads: read previous flag, write next flag
            for t in 1..n_threads.saturating_sub(1) {
                let mut thread = Thread::new(t);
                let flag_read = (n_data_locs + t - 1) as Address;
                let flag_write = (n_data_locs + t) as Address;
                // Ensure we don't exceed n_locs
                let flag_read = flag_read.min((n_locs - 1) as Address);
                let flag_write = flag_write.min((n_locs - 1) as Address);
                thread.load(0, flag_read, ord);
                thread.store(flag_write, 1, ord);
                test.add_thread(thread);
            }

            // Last thread: reads flag then reads data
            if n_threads > 1 {
                let last = n_threads - 1;
                let mut t_last = Thread::new(last);
                let last_flag = (n_data_locs + last - 1).min(n_locs - 1) as Address;
                t_last.load(0, last_flag, ord);
                for loc in 0..n_data_locs {
                    t_last.load(loc + 1, loc as Address, ord);
                }
                test.add_thread(t_last);
            }

            // Forbidden: last thread sees flag=1 but data=0
            let mut outcome = Outcome::new();
            if n_threads > 1 {
                outcome = outcome.with_reg(n_threads - 1, 0, 1); // flag = 1
                for loc in 0..n_data_locs {
                    outcome = outcome.with_reg(n_threads - 1, loc + 1, 0); // data = 0
                }
            }
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // N-thread Load Buffering (LB)
    // -----------------------------------------------------------------------

    fn gen_load_buffering(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let locs = n_locs.min(n_threads);

        for &ord in &self.config.orderings {
            let name = format!("LB-{}T-{}L-{}", n_threads, locs, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads {
                let mut thread = Thread::new(t);
                let read_loc = (t % locs) as Address;
                let write_loc = ((t + 1) % locs) as Address;
                thread.load(t, read_loc, ord);
                thread.store(write_loc, 1, ord);
                test.add_thread(thread);
            }

            let mut outcome = Outcome::new();
            for t in 0..n_threads {
                outcome = outcome.with_reg(t, t, 1);
            }
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // IRIW generalizations: N readers, M writers
    // -----------------------------------------------------------------------

    fn gen_iriw(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        if n_threads < 4 { return tests; }

        // N writers (half), N readers (half)
        let n_writers = n_threads / 2;
        let n_readers = n_threads - n_writers;
        let locs = n_locs.min(n_writers);

        for &ord in &self.config.orderings {
            let name = format!("IRIW-{}W-{}R-{}L-{}", n_writers, n_readers, locs, ord);
            let mut test = LitmusTest::new(&name);

            // Writer threads: each writes to a different location
            for w in 0..n_writers {
                let mut thread = Thread::new(w);
                thread.store((w % locs) as Address, 1, ord);
                test.add_thread(thread);
            }

            // Reader threads: each reads all written locations
            for r in 0..n_readers {
                let mut thread = Thread::new(n_writers + r);
                for loc in 0..locs {
                    thread.load(loc, loc as Address, ord);
                }
                test.add_thread(thread);
            }

            // Forbidden: readers disagree on order
            let mut outcome = Outcome::new();
            if n_readers >= 2 && locs >= 2 {
                // Reader 0 sees loc0=1, loc1=0
                // Reader 1 sees loc0=0, loc1=1
                outcome = outcome.with_reg(n_writers, 0, 1);
                outcome = outcome.with_reg(n_writers, 1, 0);
                outcome = outcome.with_reg(n_writers + 1, 0, 0);
                outcome = outcome.with_reg(n_writers + 1, 1, 1);
            }
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Write-Read Causality (WRC) chains
    // -----------------------------------------------------------------------

    fn gen_wrc(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        if n_threads < 3 { return tests; }

        for &ord in &self.config.orderings {
            let name = format!("WRC-chain-{}T-{}L-{}", n_threads, n_locs, ord);
            let mut test = LitmusTest::new(&name);

            // Thread 0: write x
            let mut t0 = Thread::new(0);
            t0.store(0, 1, ord);
            test.add_thread(t0);

            // Threads 1..N-2: read x_prev, write x_next (chaining)
            for t in 1..n_threads - 1 {
                let mut thread = Thread::new(t);
                let read_loc = ((t - 1) % n_locs) as Address;
                let write_loc = (t % n_locs) as Address;
                thread.load(0, read_loc, ord);
                thread.store(write_loc, 1, ord);
                test.add_thread(thread);
            }

            // Last thread: read x_{N-2}, read x_0
            let last = n_threads - 1;
            let mut t_last = Thread::new(last);
            t_last.load(0, ((last - 1) % n_locs) as Address, ord);
            t_last.load(1, 0, ord);
            test.add_thread(t_last);

            // Forbidden: last thread sees chain complete but x_0 = 0
            let mut outcome = Outcome::new();
            outcome = outcome.with_reg(last, 0, 1); // sees chain
            outcome = outcome.with_reg(last, 1, 0); // but not original write
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Coherence patterns (CoRR, CoWR, CoRW, CoWW) with multiple addresses
    // -----------------------------------------------------------------------

    fn gen_coherence(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();

        for &ord in &self.config.orderings {
            // CoRR: Two threads read same location, should agree on order
            for loc in 0..n_locs {
                let name = format!("CoRR-{}T-loc{}-{}", n_threads, loc, ord);
                let mut test = LitmusTest::new(&name);

                // Writer thread
                let mut tw = Thread::new(0);
                tw.store(loc as Address, 1, ord);
                tw.store(loc as Address, 2, ord);
                test.add_thread(tw);

                // Reader threads
                for r in 1..n_threads {
                    let mut thread = Thread::new(r);
                    thread.load(0, loc as Address, ord);
                    thread.load(1, loc as Address, ord);
                    test.add_thread(thread);
                }

                let mut outcome = Outcome::new();
                if n_threads > 1 {
                    outcome = outcome.with_reg(1, 0, 2); // see 2 first
                    outcome = outcome.with_reg(1, 1, 1); // then 1 — forbidden
                }
                test.expect(outcome, LitmusOutcome::Forbidden);
                tests.push(test);
            }

            // CoWW: Two writers, readers should agree on final value
            let name = format!("CoWW-{}T-{}L-{}", n_threads, n_locs, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads.min(2) {
                let mut thread = Thread::new(t);
                for loc in 0..n_locs {
                    thread.store(loc as Address, (t + 1) as Value, ord);
                }
                test.add_thread(thread);
            }
            for t in 2..n_threads {
                let mut thread = Thread::new(t);
                for loc in 0..n_locs {
                    thread.load(loc, loc as Address, ord);
                }
                test.add_thread(thread);
            }
            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Read-Modify-Write patterns
    // -----------------------------------------------------------------------

    fn gen_rmw_patterns(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();

        for &ord in &self.config.orderings {
            // CAS chain: threads do CAS in sequence
            let name = format!("CAS-chain-{}T-{}L-{}", n_threads, n_locs, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads {
                let mut thread = Thread::new(t);
                let loc = (t % n_locs) as Address;
                // RMW: atomically read old, write new
                thread.rmw(0, loc, (t + 1) as Value, ord);
                test.add_thread(thread);
            }
            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);

            // Fetch-add contention: all threads increment same location
            let name = format!("FAA-contention-{}T-{}", n_threads, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads {
                let mut thread = Thread::new(t);
                thread.rmw(0, 0, 1, ord); // fetch_add(x, 1)
                test.add_thread(thread);
            }
            // Final value should be n_threads
            let mut outcome = Outcome::new();
            outcome.memory.insert(0, n_threads as Value);
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);

            // CAS + load: lock-like pattern
            let name = format!("CAS-lock-{}T-{}L-{}", n_threads, n_locs, ord);
            let mut test = LitmusTest::new(&name);

            for t in 0..n_threads {
                let mut thread = Thread::new(t);
                let lock_loc = 0u64;
                let data_loc = (1 + (t % (n_locs - 1).max(1))) as Address;
                // Try to acquire lock
                thread.rmw(0, lock_loc, 1, ord);
                // Access data
                thread.store(data_loc, (t + 1) as Value, ord);
                thread.load(1, data_loc, ord);
                // Release lock
                thread.store(lock_loc, 0, Ordering::Release);
                test.add_thread(thread);
            }
            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Dekker's algorithm (generalized to N threads)
    // -----------------------------------------------------------------------

    fn gen_dekker(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(n_locs).min(8);

        for &ord in &self.config.orderings {
            let name = format!("Dekker-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Each thread: flag[i] = 1; if flag[j] == 0 for all j≠i: critical section
            for t in 0..n {
                let mut thread = Thread::new(t);
                // Set own flag
                thread.store(t as Address, 1, ord);
                // Fence
                thread.fence(Ordering::SeqCst, Scope::None);
                // Check other flags
                for other in 0..n {
                    if other != t {
                        thread.load(other, other as Address, ord);
                    }
                }
                test.add_thread(thread);
            }

            // Forbidden: all threads enter critical section (all reads see 0)
            let mut outcome = Outcome::new();
            for t in 0..n {
                let mut reg = 0;
                for other in 0..n {
                    if other != t {
                        outcome = outcome.with_reg(t, other, 0);
                        reg += 1;
                    }
                }
            }
            test.expect(outcome, LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Peterson's algorithm (generalized to N threads via tournament tree)
    // -----------------------------------------------------------------------

    fn gen_peterson(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(8);

        for &ord in &self.config.orderings {
            let name = format!("Peterson-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // N-thread Peterson's uses N-1 levels
            // Locations: flag[i] (0..N), turn[level] (N..N+N-1)
            let flag_base = 0u64;
            let turn_base = n as u64;

            for t in 0..n {
                let mut thread = Thread::new(t);

                // For each level
                for level in 0..(n - 1).min(3) {
                    // flag[i] = level + 1
                    thread.store(flag_base + t as u64, (level + 1) as Value, ord);
                    // turn[level] = i
                    thread.store(turn_base + level as u64, t as Value, ord);
                    // Fence
                    thread.fence(Ordering::SeqCst, Scope::None);
                    // Read turn[level]
                    thread.load(level * 2, turn_base + level as u64, ord);
                }

                // Critical section: write to shared data
                let data_loc = (turn_base + (n - 1) as u64 + 1).min((n_locs - 1) as u64);
                thread.store(data_loc, (t + 1) as Value, ord);
                thread.load(n - 1, data_loc, ord);

                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Lamport's Bakery algorithm
    // -----------------------------------------------------------------------

    fn gen_lamport_bakery(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(6); // Bakery grows quadratically

        for &ord in &self.config.orderings {
            let name = format!("Bakery-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Locations: choosing[i] (0..N), number[i] (N..2N)
            let choosing_base = 0u64;
            let number_base = n as u64;

            for t in 0..n {
                let mut thread = Thread::new(t);

                // choosing[i] = 1
                thread.store(choosing_base + t as u64, 1, ord);

                // number[i] = max(number[0..N]) + 1
                // Read all other tickets
                for j in 0..n {
                    if j != t {
                        thread.load(j, number_base + j as u64, ord);
                    }
                }
                // Write own ticket (simplified: just use thread id + 1)
                thread.store(number_base + t as u64, (t + 1) as Value, ord);

                // choosing[i] = 0
                thread.store(choosing_base + t as u64, 0, ord);

                // Fence
                thread.fence(Ordering::SeqCst, Scope::None);

                // Check each other thread (simplified)
                for j in 0..n.min(3) {
                    if j != t {
                        // Read choosing[j]
                        thread.load(n + j, choosing_base + j as u64, ord);
                        // Read number[j]
                        thread.load(n + n + j, number_base + j as u64, ord);
                    }
                }

                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Treiber stack pattern
    // -----------------------------------------------------------------------

    fn gen_treiber_stack(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(8);

        for &ord in &[Ordering::AcqRel, Ordering::SeqCst] {
            let name = format!("TreiberStack-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Location 0: head pointer, Locations 1..N: node data, N+1..2N: next pointers
            let head = 0u64;

            for t in 0..n {
                let mut thread = Thread::new(t);
                let node_data = (1 + t) as u64;
                let node_next = (1 + n + t) as u64;

                if t % 2 == 0 {
                    // Push: write data, CAS head
                    thread.store(node_data.min((n_locs - 1) as u64), (t + 1) as Value, ord);
                    thread.rmw(0, head, node_data.min((n_locs - 1) as u64), ord);
                } else {
                    // Pop: read head, read data
                    thread.load(0, head, ord);
                    thread.load(1, node_data.min((n_locs - 1) as u64), ord);
                }

                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Michael-Scott queue pattern
    // -----------------------------------------------------------------------

    fn gen_michael_scott_queue(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(8);

        for &ord in &[Ordering::AcqRel, Ordering::SeqCst] {
            let name = format!("MSQueue-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Locations: 0=head, 1=tail, 2..N+1=node data, N+2..2N+1=next ptrs
            let head_loc = 0u64;
            let tail_loc = 1u64;

            for t in 0..n {
                let mut thread = Thread::new(t);
                let node = (2 + t).min(n_locs - 1) as u64;

                if t % 2 == 0 {
                    // Enqueue
                    thread.store(node, (t + 1) as Value, Ordering::Relaxed);
                    thread.rmw(0, tail_loc, node, ord);
                } else {
                    // Dequeue
                    thread.load(0, head_loc, ord);
                    thread.rmw(1, head_loc, 0, ord);
                }

                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // Ticket lock pattern
    // -----------------------------------------------------------------------

    fn gen_ticket_lock(&self, n_threads: usize, _n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(8);

        for &ord in &[Ordering::AcqRel, Ordering::SeqCst] {
            let name = format!("TicketLock-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Locations: 0=next_ticket, 1=now_serving, 2=shared_data
            let next_ticket = 0u64;
            let now_serving = 1u64;
            let shared_data = 2u64;

            for t in 0..n {
                let mut thread = Thread::new(t);
                // my_ticket = fetch_add(next_ticket, 1)
                thread.rmw(0, next_ticket, 1, ord);
                // spin: while (now_serving != my_ticket) {}
                thread.load(1, now_serving, Ordering::Acquire);
                // critical section
                thread.store(shared_data, (t + 1) as Value, Ordering::Relaxed);
                thread.load(2, shared_data, Ordering::Relaxed);
                // unlock: now_serving++
                thread.rmw(3, now_serving, 1, Ordering::Release);
                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // MCS lock pattern
    // -----------------------------------------------------------------------

    fn gen_mcs_lock(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(6);

        for &ord in &[Ordering::AcqRel, Ordering::SeqCst] {
            let name = format!("MCSLock-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Locations: 0=tail, per-thread: locked[t], next[t]
            let tail = 0u64;
            let shared_data = 1u64;

            for t in 0..n {
                let mut thread = Thread::new(t);
                let my_locked = (2 + t * 2) as u64;
                let my_next = (3 + t * 2) as u64;
                let my_locked = my_locked.min((n_locs - 1) as u64);
                let my_next = my_next.min((n_locs - 1) as u64);

                // Initialize node
                thread.store(my_next, 0, Ordering::Relaxed);
                thread.store(my_locked, 1, Ordering::Relaxed);
                // Swap tail
                thread.rmw(0, tail, my_locked, ord);
                // Read locked (spin)
                thread.load(1, my_locked, Ordering::Acquire);
                // Critical section
                thread.store(shared_data, (t + 1) as Value, Ordering::Relaxed);
                thread.load(2, shared_data, Ordering::Relaxed);
                // Release: store next's locked = 0
                thread.store(my_locked, 0, Ordering::Release);
                test.add_thread(thread);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }

    // -----------------------------------------------------------------------
    // SeqLock pattern
    // -----------------------------------------------------------------------

    fn gen_seqlock(&self, n_threads: usize, n_locs: usize) -> Vec<LitmusTest> {
        let mut tests = Vec::new();
        let n = n_threads.min(8);

        for &ord in &[Ordering::AcqRel, Ordering::SeqCst] {
            let name = format!("SeqLock-{}T-{}", n, ord);
            let mut test = LitmusTest::new(&name);

            // Location 0: sequence counter, 1..K: data locations
            let seq_loc = 0u64;
            let n_data = (n_locs - 1).max(1);

            // Thread 0: writer
            let mut writer = Thread::new(0);
            // seq++ (odd = writing)
            writer.rmw(0, seq_loc, 1, ord);
            for d in 0..n_data {
                writer.store((1 + d) as u64, 42, Ordering::Relaxed);
            }
            // seq++ (even = done)
            writer.rmw(1, seq_loc, 1, ord);
            test.add_thread(writer);

            // Remaining threads: readers
            for t in 1..n {
                let mut reader = Thread::new(t);
                // Read seq (should be even)
                reader.load(0, seq_loc, Ordering::Acquire);
                // Read data
                for d in 0..n_data.min(3) {
                    reader.load(d + 1, (1 + d) as u64, Ordering::Relaxed);
                }
                // Read seq again (should match)
                reader.load(n_data + 1, seq_loc, Ordering::Acquire);
                test.add_thread(reader);
            }

            test.expect(Outcome::new(), LitmusOutcome::Forbidden);
            tests.push(test);
        }

        tests
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn factorial(n: u64) -> u64 {
    if n <= 1 { return 1; }
    (2..=n).fold(1u64, |acc, x| acc.saturating_mul(x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generation_produces_tests() {
        let mut gen = ScaledLitmusGenerator::with_defaults();
        let config = ScaledTestConfig {
            max_tests: 100,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        assert!(!tests.is_empty(), "Should generate at least some tests");
        assert!(tests.len() >= 10, "Should generate many tests, got {}", tests.len());
    }

    #[test]
    fn test_store_buffering_scaled() {
        let config = ScaledTestConfig {
            min_threads: 4,
            max_threads: 4,
            min_locations: 4,
            max_locations: 4,
            families: vec![PatternFamily::StoreBuffering],
            orderings: vec![Ordering::Relaxed],
            include_scoped: false,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        assert!(!tests.is_empty());
        let test = &tests[0];
        assert_eq!(test.threads.len(), 4);
    }

    #[test]
    fn test_message_passing_chain() {
        let config = ScaledTestConfig {
            min_threads: 6,
            max_threads: 6,
            min_locations: 4,
            max_locations: 4,
            families: vec![PatternFamily::MessagePassing],
            orderings: vec![Ordering::AcqRel],
            include_scoped: false,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        assert!(!tests.is_empty());
        assert_eq!(tests[0].threads.len(), 6);
    }

    #[test]
    fn test_execution_graph_estimate() {
        let config = ScaledTestConfig {
            min_threads: 4,
            max_threads: 4,
            min_locations: 3,
            max_locations: 3,
            families: vec![PatternFamily::StoreBuffering],
            orderings: vec![Ordering::Relaxed],
            include_scoped: false,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        gen.generate_all();
        let results = gen.results();
        assert!(!results.is_empty());
        // SB with 4 threads should have many execution graphs
        assert!(results[0].estimated_execution_graphs > 0);
    }

    #[test]
    fn test_dekker_n_threads() {
        let config = ScaledTestConfig {
            min_threads: 4,
            max_threads: 6,
            min_locations: 6,
            max_locations: 6,
            families: vec![PatternFamily::Dekker],
            orderings: vec![Ordering::SeqCst],
            include_scoped: false,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        assert!(tests.len() >= 3); // 4T, 5T, 6T
    }

    #[test]
    fn test_rmw_patterns() {
        let config = ScaledTestConfig {
            min_threads: 4,
            max_threads: 4,
            min_locations: 3,
            max_locations: 3,
            families: vec![PatternFamily::ReadModifyWrite],
            orderings: vec![Ordering::AcqRel],
            include_scoped: false,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        assert!(tests.len() >= 3); // CAS-chain, FAA-contention, CAS-lock
    }

    #[test]
    fn test_scoped_variants() {
        let config = ScaledTestConfig {
            min_threads: 4,
            max_threads: 4,
            min_locations: 4,
            max_locations: 4,
            families: vec![PatternFamily::StoreBuffering],
            orderings: vec![Ordering::Relaxed],
            include_scoped: true,
            ..ScaledTestConfig::default()
        };
        let mut gen = ScaledLitmusGenerator::new(config);
        let tests = gen.generate_all();
        // Should have base + 3 scoped variants (cta, gpu, sys)
        assert!(tests.len() >= 4);
    }
}
