//! Systematic litmus test generation for LITMUS∞.
//!
//! Provides template-based, constraint-based, random, coverage-guided,
//! and parameterized litmus test generation strategies.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::checker::{
    LitmusTest, Thread, Instruction, Outcome, LitmusOutcome,
    Address, Value,
};
use crate::checker::litmus::{Ordering, Scope, RegId};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy for test generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationStrategy {
    /// Enumerate all tests up to given bounds.
    Systematic,
    /// Generate from templates with parameter substitution.
    Template,
    /// Generate tests satisfying constraints.
    Constraint,
    /// Uniform random generation within bounds.
    Random,
    /// Coverage-guided generation.
    CoverageGuided,
}

impl fmt::Display for GenerationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Systematic => write!(f, "systematic"),
            Self::Template => write!(f, "template"),
            Self::Constraint => write!(f, "constraint"),
            Self::Random => write!(f, "random"),
            Self::CoverageGuided => write!(f, "coverage-guided"),
        }
    }
}

/// Configuration for the test generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Maximum number of threads.
    pub max_threads: usize,
    /// Maximum instructions per thread.
    pub max_instrs_per_thread: usize,
    /// Maximum number of memory locations.
    pub max_locations: usize,
    /// Maximum data value.
    pub max_value: Value,
    /// Which orderings to use.
    pub orderings: Vec<Ordering>,
    /// Which scopes to use.
    pub scopes: Vec<Scope>,
    /// Whether to include fences.
    pub include_fences: bool,
    /// Whether to include RMW operations.
    pub include_rmw: bool,
    /// Seed for random generation.
    pub seed: u64,
    /// Maximum number of tests to generate.
    pub max_tests: usize,
    /// Strategy to use.
    pub strategy: GenerationStrategy,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_threads: 4,
            max_instrs_per_thread: 4,
            max_locations: 3,
            max_value: 2,
            orderings: vec![
                Ordering::Relaxed,
                Ordering::Acquire,
                Ordering::Release,
                Ordering::SeqCst,
            ],
            scopes: vec![Scope::None],
            include_fences: true,
            include_rmw: true,
            seed: 42,
            max_tests: 1000,
            strategy: GenerationStrategy::Systematic,
        }
    }
}

impl GeneratorConfig {
    /// Config for GPU-aware test generation.
    pub fn gpu() -> Self {
        Self {
            scopes: vec![
                Scope::CTA,
                Scope::GPU,
                Scope::System,
            ],
            orderings: vec![
                Ordering::Relaxed,
                Ordering::AcquireCTA,
                Ordering::ReleaseCTA,
                Ordering::AcquireGPU,
                Ordering::ReleaseGPU,
                Ordering::AcquireSystem,
                Ordering::ReleaseSystem,
            ],
            ..Default::default()
        }
    }

    /// Config for minimal test generation (useful for quick checks).
    pub fn minimal() -> Self {
        Self {
            max_threads: 2,
            max_instrs_per_thread: 2,
            max_locations: 2,
            max_value: 1,
            orderings: vec![Ordering::Relaxed, Ordering::SeqCst],
            include_fences: false,
            include_rmw: false,
            max_tests: 100,
            ..Default::default()
        }
    }

    /// Address values used for generation.
    pub fn addresses(&self) -> Vec<Address> {
        (0..self.max_locations as u64).map(|i| 0x100 + i * 8).collect()
    }

    /// Data values used for generation.
    pub fn values(&self) -> Vec<Value> {
        (0..=self.max_value).collect()
    }
}

// ---------------------------------------------------------------------------
// TemplateSpec — parameterised template
// ---------------------------------------------------------------------------

/// Slot types in a template.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SlotKind {
    /// A memory ordering.
    Ordering,
    /// A memory address.
    Address,
    /// A data value.
    DataValue,
    /// A fence scope.
    FenceScope,
    /// An instruction type.
    InstructionKind,
}

/// A parameterised slot in a template.
#[derive(Debug, Clone)]
pub struct TemplateSlot {
    pub name: String,
    pub kind: SlotKind,
    pub thread: usize,
    pub instr_index: usize,
}

/// A litmus test template with parameterised slots.
#[derive(Debug, Clone)]
pub struct TemplateSpec {
    /// Template name.
    pub name: String,
    /// Base test pattern.
    pub base: LitmusTest,
    /// Parameterised slots.
    pub slots: Vec<TemplateSlot>,
    /// Description of the pattern.
    pub description: String,
}

impl TemplateSpec {
    /// Create a new template from a base test.
    pub fn new(name: &str, base: LitmusTest) -> Self {
        Self {
            name: name.into(),
            base,
            slots: Vec::new(),
            description: String::new(),
        }
    }

    /// Add a parameterised slot.
    pub fn add_slot(&mut self, name: &str, kind: SlotKind, thread: usize, instr_index: usize) {
        self.slots.push(TemplateSlot {
            name: name.into(),
            kind,
            thread,
            instr_index,
        });
    }

    /// Total number of instantiations.
    pub fn instantiation_count(&self, config: &GeneratorConfig) -> usize {
        self.slots.iter().map(|s| match s.kind {
            SlotKind::Ordering => config.orderings.len(),
            SlotKind::Address => config.addresses().len(),
            SlotKind::DataValue => config.values().len(),
            SlotKind::FenceScope => config.scopes.len(),
            SlotKind::InstructionKind => 2, // load or store
        }).product()
    }

    /// Generate all instantiations of this template.
    pub fn instantiate_all(&self, config: &GeneratorConfig) -> Vec<LitmusTest> {
        let slot_values: Vec<Vec<usize>> = self.slots.iter().map(|s| {
            let count = match s.kind {
                SlotKind::Ordering => config.orderings.len(),
                SlotKind::Address => config.addresses().len(),
                SlotKind::DataValue => config.values().len(),
                SlotKind::FenceScope => config.scopes.len(),
                SlotKind::InstructionKind => 2,
            };
            (0..count).collect()
        }).collect();

        let combos = cartesian_product(&slot_values);
        let mut tests = Vec::new();

        for combo in &combos {
            if let Some(test) = self.instantiate_one(config, combo) {
                tests.push(test);
            }
        }
        tests
    }

    /// Instantiate with specific parameter indices.
    fn instantiate_one(&self, config: &GeneratorConfig, indices: &[usize]) -> Option<LitmusTest> {
        let mut test = self.base.clone();
        let addrs = config.addresses();
        let vals = config.values();

        for (slot, &idx) in self.slots.iter().zip(indices.iter()) {
            if slot.thread >= test.threads.len() { return None; }
            let thread = &mut test.threads[slot.thread];
            if slot.instr_index >= thread.instructions.len() { return None; }

            match slot.kind {
                SlotKind::Ordering => {
                    if idx >= config.orderings.len() { return None; }
                    let ord = config.orderings[idx];
                    apply_ordering(&mut thread.instructions[slot.instr_index], ord);
                }
                SlotKind::Address => {
                    if idx >= addrs.len() { return None; }
                    apply_address(&mut thread.instructions[slot.instr_index], addrs[idx]);
                }
                SlotKind::DataValue => {
                    if idx >= vals.len() { return None; }
                    apply_value(&mut thread.instructions[slot.instr_index], vals[idx]);
                }
                SlotKind::FenceScope => {
                    if idx >= config.scopes.len() { return None; }
                    apply_scope(&mut thread.instructions[slot.instr_index], config.scopes[idx]);
                }
                SlotKind::InstructionKind => {
                    // 0 = load, 1 = store
                    // skip — just keep the base instruction
                }
            }
        }

        // Update name to reflect parameterisation.
        let suffix: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
        test.name = format!("{}-{}", self.name, suffix.join("_"));
        Some(test)
    }
}

// ---------------------------------------------------------------------------
// ConstraintSpec
// ---------------------------------------------------------------------------

/// A constraint on generated tests.
#[derive(Debug, Clone)]
pub enum ConstraintSpec {
    /// Must have at least N threads.
    MinThreads(usize),
    /// Must access at least N locations.
    MinLocations(usize),
    /// Must include a fence.
    RequiresFence,
    /// Must include an RMW.
    RequiresRMW,
    /// Must include cross-thread communication (shared address).
    RequiresCommunication,
    /// Must have a specific pattern structure.
    RequiresPattern(PatternConstraint),
    /// Custom predicate.
    Custom(String),
}

/// Pattern structure constraints.
#[derive(Debug, Clone)]
pub enum PatternConstraint {
    /// Write-then-read on same location across threads.
    MessagePassing,
    /// Two threads each store then load different locations.
    StoreBuffering,
    /// Two threads each load then store different locations.
    LoadBuffering,
    /// Two writes to same location from different threads observed differently.
    CoherenceConflict,
    /// Four threads: two writers, two readers observing different orders.
    IRIW,
}

impl ConstraintSpec {
    /// Check whether a test satisfies this constraint.
    pub fn satisfied_by(&self, test: &LitmusTest) -> bool {
        match self {
            Self::MinThreads(n) => test.thread_count() >= *n,
            Self::MinLocations(n) => test.all_addresses().len() >= *n,
            Self::RequiresFence => test.threads.iter().any(|t|
                t.instructions.iter().any(|i| matches!(i, Instruction::Fence { .. }))
            ),
            Self::RequiresRMW => test.threads.iter().any(|t|
                t.instructions.iter().any(|i| matches!(i, Instruction::RMW { .. }))
            ),
            Self::RequiresCommunication => {
                let mut addr_threads: HashMap<Address, HashSet<usize>> = HashMap::new();
                for t in &test.threads {
                    for addr in t.accessed_addresses() {
                        addr_threads.entry(addr).or_default().insert(t.id);
                    }
                }
                addr_threads.values().any(|threads| threads.len() > 1)
            }
            Self::RequiresPattern(pat) => check_pattern(test, pat),
            Self::Custom(_) => true,
        }
    }
}

/// Check whether a test matches a pattern constraint.
fn check_pattern(test: &LitmusTest, pattern: &PatternConstraint) -> bool {
    match pattern {
        PatternConstraint::MessagePassing => {
            if test.thread_count() < 2 { return false; }
            let t0 = &test.threads[0];
            let t1 = &test.threads[1];
            let has_writes = t0.instructions.iter().filter(|i| matches!(i, Instruction::Store { .. })).count() >= 2;
            let has_reads = t1.instructions.iter().filter(|i| matches!(i, Instruction::Load { .. })).count() >= 2;
            has_writes && has_reads
        }
        PatternConstraint::StoreBuffering => {
            if test.thread_count() < 2 { return false; }
            test.threads.iter().all(|t| {
                let has_store = t.instructions.iter().any(|i| matches!(i, Instruction::Store { .. }));
                let has_load = t.instructions.iter().any(|i| matches!(i, Instruction::Load { .. }));
                has_store && has_load
            })
        }
        PatternConstraint::LoadBuffering => {
            if test.thread_count() < 2 { return false; }
            for t in &test.threads {
                let first_is_load = t.instructions.first().map_or(false, |i| matches!(i, Instruction::Load { .. }));
                if !first_is_load { return false; }
            }
            true
        }
        PatternConstraint::CoherenceConflict => {
            let mut write_addrs: HashMap<Address, usize> = HashMap::new();
            for t in &test.threads {
                for i in &t.instructions {
                    if let Instruction::Store { addr, .. } = i {
                        *write_addrs.entry(*addr).or_default() += 1;
                    }
                }
            }
            write_addrs.values().any(|&count| count >= 2)
        }
        PatternConstraint::IRIW => {
            test.thread_count() >= 4
        }
    }
}

// ---------------------------------------------------------------------------
// TestFamily — parameterised test families
// ---------------------------------------------------------------------------

/// A family of related litmus tests with shared structure.
#[derive(Debug, Clone)]
pub struct TestFamily {
    /// Family name (e.g. "MP", "SB", "LB").
    pub name: String,
    /// Description of the family.
    pub description: String,
    /// The parameter axes for this family.
    pub axes: Vec<FamilyAxis>,
    /// The base test.
    pub base: LitmusTest,
}

/// One axis of parameterisation in a test family.
#[derive(Debug, Clone)]
pub struct FamilyAxis {
    /// Axis name.
    pub name: String,
    /// The orderings this axis ranges over.
    pub orderings: Vec<Ordering>,
}

impl TestFamily {
    /// Create a new test family.
    pub fn new(name: &str, base: LitmusTest) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            axes: Vec::new(),
            base,
        }
    }

    /// Add an axis of parameterisation.
    pub fn add_axis(&mut self, name: &str, orderings: Vec<Ordering>) {
        self.axes.push(FamilyAxis {
            name: name.into(),
            orderings,
        });
    }

    /// Total number of members in this family.
    pub fn size(&self) -> usize {
        if self.axes.is_empty() { return 1; }
        self.axes.iter().map(|a| a.orderings.len()).product()
    }

    /// Generate all members of this family.
    pub fn generate_all(&self) -> Vec<LitmusTest> {
        if self.axes.is_empty() {
            return vec![self.base.clone()];
        }

        let dims: Vec<usize> = self.axes.iter().map(|a| a.orderings.len()).collect();
        let combos = enumerate_indices(&dims);
        let mut tests = Vec::new();

        for combo in &combos {
            let mut test = self.base.clone();
            let mut name_parts = vec![self.name.clone()];

            for (axis_idx, &ord_idx) in combo.iter().enumerate() {
                let ord = self.axes[axis_idx].orderings[ord_idx];
                name_parts.push(format!("{}", ord));
            }
            test.name = name_parts.join("-");
            tests.push(test);
        }
        tests
    }
}

/// Build the MP (Message Passing) family.
pub fn mp_family() -> TestFamily {
    let base = crate::checker::litmus::mp_test();
    let mut family = TestFamily::new("MP", base);
    family.description = "Message Passing variants".into();
    family.add_axis("w-ordering", vec![
        Ordering::Relaxed, Ordering::Release, Ordering::SeqCst,
    ]);
    family.add_axis("r-ordering", vec![
        Ordering::Relaxed, Ordering::Acquire, Ordering::SeqCst,
    ]);
    family
}

/// Build the SB (Store Buffering) family.
pub fn sb_family() -> TestFamily {
    let base = crate::checker::litmus::sb_test();
    let mut family = TestFamily::new("SB", base);
    family.description = "Store Buffering variants".into();
    family.add_axis("ordering", vec![
        Ordering::Relaxed, Ordering::SeqCst,
    ]);
    family
}

/// Build the LB (Load Buffering) family.
pub fn lb_family() -> TestFamily {
    let base = crate::checker::litmus::lb_test();
    let mut family = TestFamily::new("LB", base);
    family.description = "Load Buffering variants".into();
    family.add_axis("ordering", vec![
        Ordering::Relaxed, Ordering::Acquire, Ordering::Release, Ordering::SeqCst,
    ]);
    family
}

// ---------------------------------------------------------------------------
// TestGenerator — main generator
// ---------------------------------------------------------------------------

/// Main litmus test generator.
#[derive(Debug)]
pub struct TestGenerator {
    config: GeneratorConfig,
    /// Generated tests.
    tests: Vec<LitmusTest>,
    /// Templates registered for instantiation.
    templates: Vec<TemplateSpec>,
    /// Constraints that generated tests must satisfy.
    constraints: Vec<ConstraintSpec>,
    /// Families registered for expansion.
    families: Vec<TestFamily>,
    /// Simple RNG state.
    rng_state: u64,
}

impl TestGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        let seed = config.seed;
        Self {
            config,
            tests: Vec::new(),
            templates: Vec::new(),
            constraints: Vec::new(),
            families: Vec::new(),
            rng_state: seed,
        }
    }

    /// Create a generator with default configuration.
    pub fn default_generator() -> Self {
        Self::new(GeneratorConfig::default())
    }

    /// Register a template.
    pub fn add_template(&mut self, template: TemplateSpec) {
        self.templates.push(template);
    }

    /// Register a constraint.
    pub fn add_constraint(&mut self, constraint: ConstraintSpec) {
        self.constraints.push(constraint);
    }

    /// Register a test family.
    pub fn add_family(&mut self, family: TestFamily) {
        self.families.push(family);
    }

    /// Get generated tests.
    pub fn tests(&self) -> &[LitmusTest] {
        &self.tests
    }

    /// Consume and return generated tests.
    pub fn into_tests(self) -> Vec<LitmusTest> {
        self.tests
    }

    /// Generate tests using the configured strategy.
    pub fn generate(&mut self) -> usize {
        match self.config.strategy {
            GenerationStrategy::Systematic => self.generate_systematic(),
            GenerationStrategy::Template => self.generate_from_templates(),
            GenerationStrategy::Constraint => self.generate_constrained(),
            GenerationStrategy::Random => self.generate_random(),
            GenerationStrategy::CoverageGuided => self.generate_coverage_guided(),
        }
    }

    // -----------------------------------------------------------------------
    // Systematic generation
    // -----------------------------------------------------------------------

    /// Systematically enumerate litmus tests up to configuration bounds.
    pub fn generate_systematic(&mut self) -> usize {
        let mut count = 0;
        let addrs = self.config.addresses();
        let max_val = self.config.max_value;

        // Generate 2-thread tests with varying instruction counts.
        for n_instrs_t0 in 1..=self.config.max_instrs_per_thread.min(3) {
            for n_instrs_t1 in 1..=self.config.max_instrs_per_thread.min(3) {
                let tests = self.enumerate_two_thread_tests(
                    n_instrs_t0, n_instrs_t1, &addrs, max_val,
                );
                for test in tests {
                    if self.check_constraints(&test) {
                        self.tests.push(test);
                        count += 1;
                        if count >= self.config.max_tests { return count; }
                    }
                }
            }
        }
        count
    }

    /// Enumerate all 2-thread tests with given instruction counts.
    fn enumerate_two_thread_tests(
        &self,
        n0: usize,
        n1: usize,
        addrs: &[Address],
        max_val: Value,
    ) -> Vec<LitmusTest> {
        let instrs0 = self.enumerate_instruction_sequences(n0, addrs, max_val, 0);
        let instrs1 = self.enumerate_instruction_sequences(n1, addrs, max_val, 0);

        let mut tests = Vec::new();
        let limit = self.config.max_tests.saturating_sub(self.tests.len());

        for (i, seq0) in instrs0.iter().enumerate() {
            for (j, seq1) in instrs1.iter().enumerate() {
                if tests.len() >= limit { return tests; }

                let mut test = LitmusTest::new(&format!("sys-{}-{}", i, j));
                for &addr in addrs {
                    test.set_initial(addr, 0);
                }
                test.add_thread(Thread::with_instructions(0, seq0.clone()));
                test.add_thread(Thread::with_instructions(1, seq1.clone()));
                tests.push(test);
            }
        }
        tests
    }

    /// Enumerate instruction sequences of a given length.
    fn enumerate_instruction_sequences(
        &self,
        len: usize,
        addrs: &[Address],
        max_val: Value,
        _thread_id: usize,
    ) -> Vec<Vec<Instruction>> {
        if len == 0 { return vec![vec![]]; }

        let base_instrs = self.enumerate_single_instructions(addrs, max_val);
        if len == 1 {
            return base_instrs.into_iter().map(|i| vec![i]).collect();
        }

        let shorter = self.enumerate_instruction_sequences(len - 1, addrs, max_val, _thread_id);
        let mut result = Vec::new();
        let limit = 200; // bound combinatorial explosion

        for seq in &shorter {
            for instr in &base_instrs {
                if result.len() >= limit { return result; }
                let mut new_seq = seq.clone();
                new_seq.push(instr.clone());
                result.push(new_seq);
            }
        }
        result
    }

    /// Enumerate single instructions.
    fn enumerate_single_instructions(
        &self,
        addrs: &[Address],
        max_val: Value,
    ) -> Vec<Instruction> {
        let mut instrs = Vec::new();

        for &ord in &self.config.orderings {
            for &addr in addrs {
                // Load r0 from addr
                instrs.push(Instruction::Load { reg: 0, addr, ordering: ord });

                // Store values to addr
                for val in 1..=max_val {
                    instrs.push(Instruction::Store { addr, value: val, ordering: ord });
                }
            }
        }

        if self.config.include_fences {
            for &scope in &self.config.scopes {
                instrs.push(Instruction::Fence {
                    ordering: Ordering::SeqCst,
                    scope,
                });
            }
        }

        if self.config.include_rmw {
            for &ord in &self.config.orderings {
                for &addr in addrs {
                    instrs.push(Instruction::RMW { reg: 0, addr, value: 1, ordering: ord });
                }
            }
        }

        instrs
    }

    // -----------------------------------------------------------------------
    // Template-based generation
    // -----------------------------------------------------------------------

    /// Generate tests from registered templates.
    pub fn generate_from_templates(&mut self) -> usize {
        let mut count = 0;
        let templates: Vec<TemplateSpec> = self.templates.clone();
        for template in &templates {
            let tests = template.instantiate_all(&self.config);
            for test in tests {
                if count >= self.config.max_tests { return count; }
                if self.check_constraints(&test) {
                    self.tests.push(test);
                    count += 1;
                }
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Constraint-based generation
    // -----------------------------------------------------------------------

    /// Generate tests that satisfy all registered constraints.
    pub fn generate_constrained(&mut self) -> usize {
        // First generate a pool, then filter.
        let saved_strategy = self.config.strategy;
        self.config.strategy = GenerationStrategy::Systematic;
        let pool_size = self.config.max_tests * 5;
        let saved_max = self.config.max_tests;
        self.config.max_tests = pool_size;

        let mut pool = Vec::new();
        let addrs = self.config.addresses();
        let max_val = self.config.max_value;

        for n0 in 1..=self.config.max_instrs_per_thread.min(2) {
            for n1 in 1..=self.config.max_instrs_per_thread.min(2) {
                let tests = self.enumerate_two_thread_tests(n0, n1, &addrs, max_val);
                pool.extend(tests);
                if pool.len() >= pool_size { break; }
            }
            if pool.len() >= pool_size { break; }
        }

        self.config.strategy = saved_strategy;
        self.config.max_tests = saved_max;

        let mut count = 0;
        for test in pool {
            if count >= self.config.max_tests { break; }
            if self.check_constraints(&test) {
                self.tests.push(test);
                count += 1;
            }
        }
        count
    }

    // -----------------------------------------------------------------------
    // Random generation
    // -----------------------------------------------------------------------

    /// Generate random litmus tests.
    pub fn generate_random(&mut self) -> usize {
        let mut count = 0;
        let addrs = self.config.addresses();
        let max_val = self.config.max_value;

        while count < self.config.max_tests {
            let n_threads = 2 + (self.next_rand() % (self.config.max_threads as u64 - 1)) as usize;
            let n_threads = n_threads.min(self.config.max_threads);

            let mut test = LitmusTest::new(&format!("rnd-{}", count));
            for &addr in &addrs {
                test.set_initial(addr, 0);
            }

            for tid in 0..n_threads {
                let n_instrs = 1 + (self.next_rand() % self.config.max_instrs_per_thread as u64) as usize;
                let thread = self.generate_random_thread(tid, n_instrs, &addrs, max_val);
                test.add_thread(thread);
            }

            if self.check_constraints(&test) {
                self.tests.push(test);
                count += 1;
            }
        }
        count
    }

    /// Generate a random thread.
    fn generate_random_thread(
        &mut self,
        id: usize,
        n_instrs: usize,
        addrs: &[Address],
        max_val: Value,
    ) -> Thread {
        let mut thread = Thread::new(id);
        let mut reg_counter = 0usize;

        for _ in 0..n_instrs {
            let kind = self.next_rand() % 4;
            let addr_idx = (self.next_rand() % addrs.len() as u64) as usize;
            let addr = addrs[addr_idx];
            let ord_idx = (self.next_rand() % self.config.orderings.len() as u64) as usize;
            let ord = self.config.orderings[ord_idx];

            match kind {
                0 => {
                    // Load
                    thread.load(reg_counter, addr, ord);
                    reg_counter += 1;
                }
                1 => {
                    // Store
                    let val = 1 + (self.next_rand() % max_val);
                    thread.store(addr, val, ord);
                }
                2 if self.config.include_fences => {
                    let scope_idx = (self.next_rand() % self.config.scopes.len() as u64) as usize;
                    thread.fence(Ordering::SeqCst, self.config.scopes[scope_idx]);
                }
                3 if self.config.include_rmw => {
                    let val = 1 + (self.next_rand() % max_val);
                    thread.rmw(reg_counter, addr, val, ord);
                    reg_counter += 1;
                }
                _ => {
                    // Default to load.
                    thread.load(reg_counter, addr, ord);
                    reg_counter += 1;
                }
            }
        }
        thread
    }

    // -----------------------------------------------------------------------
    // Coverage-guided generation
    // -----------------------------------------------------------------------

    /// Generate tests guided by coverage metrics.
    pub fn generate_coverage_guided(&mut self) -> usize {
        // Generate a large random pool, then select for coverage diversity.
        let saved = self.config.max_tests;
        self.config.max_tests = saved * 10;
        self.generate_random();
        self.config.max_tests = saved;

        // Sort by structural diversity.
        let all_tests = std::mem::take(&mut self.tests);
        let selected = select_diverse(all_tests, saved);
        self.tests = selected;
        self.tests.len()
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Check that a test satisfies all registered constraints.
    fn check_constraints(&self, test: &LitmusTest) -> bool {
        self.constraints.iter().all(|c| c.satisfied_by(test))
    }

    /// Simple xorshift64 PRNG.
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

// ---------------------------------------------------------------------------
// Diversity selection
// ---------------------------------------------------------------------------

/// Signature for structural diversity measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TestSignature {
    n_threads: usize,
    n_loads: usize,
    n_stores: usize,
    n_fences: usize,
    n_rmw: usize,
    n_locations: usize,
    has_shared_location: bool,
}

fn compute_signature(test: &LitmusTest) -> TestSignature {
    let mut n_loads = 0;
    let mut n_stores = 0;
    let mut n_fences = 0;
    let mut n_rmw = 0;

    for t in &test.threads {
        for i in &t.instructions {
            match i {
                Instruction::Load { .. } => n_loads += 1,
                Instruction::Store { .. } => n_stores += 1,
                Instruction::Fence { .. } => n_fences += 1,
                Instruction::RMW { .. } => n_rmw += 1,
                _ => {}
            }
        }
    }

    let addrs = test.all_addresses();
    let mut addr_threads: HashMap<Address, HashSet<usize>> = HashMap::new();
    for t in &test.threads {
        for addr in t.accessed_addresses() {
            addr_threads.entry(addr).or_default().insert(t.id);
        }
    }
    let has_shared = addr_threads.values().any(|ts| ts.len() > 1);

    TestSignature {
        n_threads: test.thread_count(),
        n_loads,
        n_stores,
        n_fences,
        n_rmw,
        n_locations: addrs.len(),
        has_shared_location: has_shared,
    }
}

/// Select a diverse subset of tests based on structural signatures.
fn select_diverse(tests: Vec<LitmusTest>, max: usize) -> Vec<LitmusTest> {
    let mut seen_sigs: HashSet<TestSignature> = HashSet::new();
    let mut selected = Vec::new();

    // First pass: one test per unique signature.
    for test in &tests {
        if selected.len() >= max { break; }
        let sig = compute_signature(test);
        if seen_sigs.insert(sig) {
            selected.push(test.clone());
        }
    }

    // Second pass: fill remaining slots.
    if selected.len() < max {
        for test in &tests {
            if selected.len() >= max { break; }
            // Simple dedup by name
            if !selected.iter().any(|t| t.name == test.name) {
                selected.push(test.clone());
            }
        }
    }

    selected
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the Cartesian product of index ranges.
fn cartesian_product(ranges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if ranges.is_empty() { return vec![vec![]]; }

    let mut result = vec![vec![]];
    for range in ranges {
        let mut new_result = Vec::new();
        for combo in &result {
            for &val in range {
                let mut new_combo = combo.clone();
                new_combo.push(val);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    result
}

/// Enumerate all index combinations for given dimension sizes.
fn enumerate_indices(dims: &[usize]) -> Vec<Vec<usize>> {
    if dims.is_empty() { return vec![vec![]]; }

    let ranges: Vec<Vec<usize>> = dims.iter().map(|&d| (0..d).collect()).collect();
    cartesian_product(&ranges)
}

/// Apply an ordering to an instruction (if applicable).
fn apply_ordering(instr: &mut Instruction, ord: Ordering) {
    match instr {
        Instruction::Load { ordering, .. } => *ordering = ord,
        Instruction::Store { ordering, .. } => *ordering = ord,
        Instruction::Fence { ordering, .. } => *ordering = ord,
        Instruction::RMW { ordering, .. } => *ordering = ord,
        _ => {}
    }
}

/// Apply an address to an instruction (if applicable).
fn apply_address(instr: &mut Instruction, addr: Address) {
    match instr {
        Instruction::Load { addr: a, .. } => *a = addr,
        Instruction::Store { addr: a, .. } => *a = addr,
        Instruction::RMW { addr: a, .. } => *a = addr,
        _ => {}
    }
}

/// Apply a value to an instruction (if applicable).
fn apply_value(instr: &mut Instruction, val: Value) {
    match instr {
        Instruction::Store { value, .. } => *value = val,
        Instruction::RMW { value, .. } => *value = val,
        _ => {}
    }
}

/// Apply a scope to a fence instruction.
fn apply_scope(instr: &mut Instruction, s: Scope) {
    if let Instruction::Fence { scope, .. } = instr {
        *scope = s;
    }
}

// ---------------------------------------------------------------------------
// Predefined template builders
// ---------------------------------------------------------------------------

/// Build a Message Passing template with ordering slots.
pub fn mp_template() -> TemplateSpec {
    let base = crate::checker::litmus::mp_test();
    let mut tmpl = TemplateSpec::new("MP-tmpl", base);
    tmpl.description = "Message Passing with parameterised orderings".into();
    tmpl.add_slot("w1-ord", SlotKind::Ordering, 0, 0);
    tmpl.add_slot("w2-ord", SlotKind::Ordering, 0, 1);
    tmpl.add_slot("r1-ord", SlotKind::Ordering, 1, 0);
    tmpl.add_slot("r2-ord", SlotKind::Ordering, 1, 1);
    tmpl
}

/// Build a Store Buffering template with ordering slots.
pub fn sb_template() -> TemplateSpec {
    let base = crate::checker::litmus::sb_test();
    let mut tmpl = TemplateSpec::new("SB-tmpl", base);
    tmpl.description = "Store Buffering with parameterised orderings".into();
    tmpl.add_slot("t0-w-ord", SlotKind::Ordering, 0, 0);
    tmpl.add_slot("t0-r-ord", SlotKind::Ordering, 0, 1);
    tmpl.add_slot("t1-w-ord", SlotKind::Ordering, 1, 0);
    tmpl.add_slot("t1-r-ord", SlotKind::Ordering, 1, 1);
    tmpl
}

/// Build a Load Buffering template with ordering slots.
pub fn lb_template() -> TemplateSpec {
    let base = crate::checker::litmus::lb_test();
    let mut tmpl = TemplateSpec::new("LB-tmpl", base);
    tmpl.description = "Load Buffering with parameterised orderings".into();
    tmpl.add_slot("t0-r-ord", SlotKind::Ordering, 0, 0);
    tmpl.add_slot("t0-w-ord", SlotKind::Ordering, 0, 1);
    tmpl.add_slot("t1-r-ord", SlotKind::Ordering, 1, 0);
    tmpl.add_slot("t1-w-ord", SlotKind::Ordering, 1, 1);
    tmpl
}

/// Build a 2+2W (coherence) template.
pub fn two_plus_two_w_template() -> TemplateSpec {
    let base = crate::checker::litmus::two_plus_two_w_test();
    let mut tmpl = TemplateSpec::new("2+2W-tmpl", base);
    tmpl.description = "2+2W (coherence) with parameterised orderings".into();
    tmpl.add_slot("t0-w1-ord", SlotKind::Ordering, 0, 0);
    tmpl.add_slot("t0-w2-ord", SlotKind::Ordering, 0, 1);
    tmpl.add_slot("t1-w1-ord", SlotKind::Ordering, 1, 0);
    tmpl.add_slot("t1-w2-ord", SlotKind::Ordering, 1, 1);
    tmpl
}

// ---------------------------------------------------------------------------
// GPU-specific template builders
// ---------------------------------------------------------------------------

/// Generate GPU-scoped MP tests across scope combinations.
pub fn gpu_mp_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let scopes = &config.scopes;
    let mut tests = Vec::new();

    let x: Address = 0x100;
    let y: Address = 0x108;

    for &w_scope in scopes {
        for &r_scope in scopes {
            let w_ord = scope_to_release(w_scope);
            let r_ord = scope_to_acquire(r_scope);

            let mut test = LitmusTest::new(&format!(
                "MP-gpu-{}-{}", w_scope, r_scope
            ));
            test.set_initial(x, 0);
            test.set_initial(y, 0);

            let mut t0 = Thread::new(0);
            t0.store(x, 1, w_ord);
            t0.store(y, 1, w_ord);
            test.add_thread(t0);

            let mut t1 = Thread::new(1);
            t1.load(0, y, r_ord);
            t1.load(1, x, r_ord);
            test.add_thread(t1);

            test.expect(
                Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
                LitmusOutcome::Forbidden,
            );

            tests.push(test);
        }
    }
    tests
}

/// Generate GPU-scoped SB tests across scope combinations.
pub fn gpu_sb_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let scopes = &config.scopes;
    let mut tests = Vec::new();

    let x: Address = 0x100;
    let y: Address = 0x108;

    for &scope in scopes {
        let w_ord = scope_to_release(scope);
        let r_ord = scope_to_acquire(scope);

        let mut test = LitmusTest::new(&format!("SB-gpu-{}", scope));
        test.set_initial(x, 0);
        test.set_initial(y, 0);

        let mut t0 = Thread::new(0);
        t0.store(x, 1, w_ord);
        t0.load(0, y, r_ord);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(y, 1, w_ord);
        t1.load(0, x, r_ord);
        test.add_thread(t1);

        test.expect(
            Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
            LitmusOutcome::Forbidden,
        );

        tests.push(test);
    }
    tests
}

/// Generate fence-variant tests: insert fences between instructions.
pub fn fence_variants(base: &LitmusTest, config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();

    for &scope in &config.scopes {
        for tid in 0..base.thread_count() {
            let thread = &base.threads[tid];
            // Insert a fence between every pair of consecutive instructions.
            for pos in 0..thread.instructions.len().saturating_sub(1) {
                let mut test = base.clone();
                let fence = Instruction::Fence {
                    ordering: Ordering::SeqCst,
                    scope,
                };
                test.threads[tid].instructions.insert(pos + 1, fence);
                test.name = format!("{}-fence-T{}-{}-{}", base.name, tid, pos, scope);
                tests.push(test);
            }
        }
    }
    tests
}

/// Map a scope to a release ordering.
fn scope_to_release(scope: Scope) -> Ordering {
    match scope {
        Scope::CTA => Ordering::ReleaseCTA,
        Scope::GPU => Ordering::ReleaseGPU,
        Scope::System => Ordering::ReleaseSystem,
        Scope::None => Ordering::Release,
    }
}

/// Map a scope to an acquire ordering.
fn scope_to_acquire(scope: Scope) -> Ordering {
    match scope {
        Scope::CTA => Ordering::AcquireCTA,
        Scope::GPU => Ordering::AcquireGPU,
        Scope::System => Ordering::AcquireSystem,
        Scope::None => Ordering::Acquire,
    }
}

// ---------------------------------------------------------------------------
// Multi-threaded patterns
// ---------------------------------------------------------------------------

/// Generate 3-thread WRC (Write-Read-Coherence) variants.
pub fn wrc_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();
    let x: Address = 0x100;
    let y: Address = 0x108;

    for &w_ord in &config.orderings {
        for &r_ord in &config.orderings {
            if !w_ord.is_release() && w_ord != Ordering::SeqCst && w_ord != Ordering::Relaxed {
                continue;
            }

            let mut test = LitmusTest::new(&format!("WRC-{}-{}", w_ord, r_ord));
            test.set_initial(x, 0);
            test.set_initial(y, 0);

            // T0: W(x)=1
            let mut t0 = Thread::new(0);
            t0.store(x, 1, w_ord);
            test.add_thread(t0);

            // T1: R(x)=1; W(y)=1
            let mut t1 = Thread::new(1);
            t1.load(0, x, r_ord);
            t1.store(y, 1, w_ord);
            test.add_thread(t1);

            // T2: R(y)=1; R(x)=?
            let mut t2 = Thread::new(2);
            t2.load(0, y, r_ord);
            t2.load(1, x, r_ord);
            test.add_thread(t2);

            // Forbidden: T1:r0=1, T2:r0=1, T2:r1=0
            test.expect(
                Outcome::new()
                    .with_reg(1, 0, 1)
                    .with_reg(2, 0, 1)
                    .with_reg(2, 1, 0),
                LitmusOutcome::Forbidden,
            );

            tests.push(test);
        }
    }
    tests
}

/// Generate ISA2 variants.
pub fn isa2_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();
    let x: Address = 0x100;
    let y: Address = 0x108;
    let z: Address = 0x110;

    for &ord in &config.orderings {
        let mut test = LitmusTest::new(&format!("ISA2-{}", ord));
        test.set_initial(x, 0);
        test.set_initial(y, 0);
        test.set_initial(z, 0);

        // T0: W(x)=1; W(y)=1
        let mut t0 = Thread::new(0);
        t0.store(x, 1, ord);
        t0.store(y, 1, ord);
        test.add_thread(t0);

        // T1: R(y)=1; W(z)=1
        let mut t1 = Thread::new(1);
        t1.load(0, y, ord);
        t1.store(z, 1, ord);
        test.add_thread(t1);

        // T2: R(z)=1; R(x)=?
        let mut t2 = Thread::new(2);
        t2.load(0, z, ord);
        t2.load(1, x, ord);
        test.add_thread(t2);

        // Forbidden: T1:r0=1, T2:r0=1, T2:r1=0
        test.expect(
            Outcome::new()
                .with_reg(1, 0, 1)
                .with_reg(2, 0, 1)
                .with_reg(2, 1, 0),
            LitmusOutcome::Forbidden,
        );

        tests.push(test);
    }
    tests
}

/// Generate R (read-read coherence) pattern.
pub fn r_pattern_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();
    let x: Address = 0x100;

    for &ord in &config.orderings {
        let mut test = LitmusTest::new(&format!("R-{}", ord));
        test.set_initial(x, 0);

        // T0: W(x)=1; W(x)=2
        let mut t0 = Thread::new(0);
        t0.store(x, 1, ord);
        t0.store(x, 2, ord);
        test.add_thread(t0);

        // T1: R(x)=?; R(x)=?
        let mut t1 = Thread::new(1);
        t1.load(0, x, ord);
        t1.load(1, x, ord);
        test.add_thread(t1);

        // Forbidden: T1:r0=2, T1:r1=1 (violated coherence)
        test.expect(
            Outcome::new().with_reg(1, 0, 2).with_reg(1, 1, 1),
            LitmusOutcome::Forbidden,
        );

        tests.push(test);
    }
    tests
}

/// Generate S (store atomicity) pattern.
pub fn s_pattern_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();
    let x: Address = 0x100;

    for &ord in &config.orderings {
        let mut test = LitmusTest::new(&format!("S-{}", ord));
        test.set_initial(x, 0);

        // T0: W(x)=1
        let mut t0 = Thread::new(0);
        t0.store(x, 1, ord);
        test.add_thread(t0);

        // T1: W(x)=2
        let mut t1 = Thread::new(1);
        t1.store(x, 2, ord);
        test.add_thread(t1);

        // T2: R(x)=?; R(x)=?
        let mut t2 = Thread::new(2);
        t2.load(0, x, ord);
        t2.load(1, x, ord);
        test.add_thread(t2);

        tests.push(test);
    }
    tests
}

// ---------------------------------------------------------------------------
// RMW-based patterns
// ---------------------------------------------------------------------------

/// Generate tests with atomic RMW operations.
pub fn rmw_variants(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut tests = Vec::new();
    let x: Address = 0x100;
    let y: Address = 0x108;

    for &ord in &config.orderings {
        // RMW + load pattern (lock-like).
        let mut test = LitmusTest::new(&format!("RMW-lock-{}", ord));
        test.set_initial(x, 0);
        test.set_initial(y, 0);

        let mut t0 = Thread::new(0);
        t0.rmw(0, x, 1, ord);
        t0.store(y, 1, ord);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, y, ord);
        t1.load(1, x, ord);
        test.add_thread(t1);

        tests.push(test);

        // Double RMW pattern.
        let mut test2 = LitmusTest::new(&format!("RMW-double-{}", ord));
        test2.set_initial(x, 0);

        let mut t0 = Thread::new(0);
        t0.rmw(0, x, 1, ord);
        test2.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.rmw(0, x, 2, ord);
        test2.add_thread(t1);

        tests.push(test2);
    }
    tests
}

// ---------------------------------------------------------------------------
// Batch generation
// ---------------------------------------------------------------------------

/// Generate a comprehensive test suite combining multiple strategies.
pub fn generate_comprehensive_suite(config: &GeneratorConfig) -> Vec<LitmusTest> {
    let mut all_tests = Vec::new();

    // Template-based variants.
    let mp = mp_template();
    all_tests.extend(mp.instantiate_all(config));

    let sb = sb_template();
    all_tests.extend(sb.instantiate_all(config));

    // Family-based variants.
    let mp_fam = mp_family();
    all_tests.extend(mp_fam.generate_all());

    let sb_fam = sb_family();
    all_tests.extend(sb_fam.generate_all());

    let lb_fam = lb_family();
    all_tests.extend(lb_fam.generate_all());

    // Multi-thread patterns.
    all_tests.extend(wrc_variants(config));
    all_tests.extend(isa2_variants(config));
    all_tests.extend(r_pattern_variants(config));
    all_tests.extend(s_pattern_variants(config));

    // RMW patterns.
    all_tests.extend(rmw_variants(config));

    // Fence variants of base tests.
    let base_mp = crate::checker::litmus::mp_test();
    all_tests.extend(fence_variants(&base_mp, config));

    let base_sb = crate::checker::litmus::sb_test();
    all_tests.extend(fence_variants(&base_sb, config));

    // GPU variants (if scoped).
    if config.scopes.iter().any(|s| *s != Scope::None) {
        all_tests.extend(gpu_mp_variants(config));
        all_tests.extend(gpu_sb_variants(config));
    }

    // Truncate to max.
    all_tests.truncate(config.max_tests);
    all_tests
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.max_threads, 4);
        assert_eq!(config.max_instrs_per_thread, 4);
        assert_eq!(config.max_locations, 3);
        assert!(config.include_fences);
    }

    #[test]
    fn test_generator_config_gpu() {
        let config = GeneratorConfig::gpu();
        assert!(config.scopes.contains(&Scope::CTA));
        assert!(config.scopes.contains(&Scope::GPU));
        assert!(config.scopes.contains(&Scope::System));
    }

    #[test]
    fn test_generator_config_minimal() {
        let config = GeneratorConfig::minimal();
        assert_eq!(config.max_threads, 2);
        assert_eq!(config.max_instrs_per_thread, 2);
        assert!(!config.include_fences);
    }

    #[test]
    fn test_addresses() {
        let config = GeneratorConfig { max_locations: 3, ..Default::default() };
        let addrs = config.addresses();
        assert_eq!(addrs.len(), 3);
    }

    #[test]
    fn test_values() {
        let config = GeneratorConfig { max_value: 2, ..Default::default() };
        let vals = config.values();
        assert_eq!(vals, vec![0, 1, 2]);
    }

    #[test]
    fn test_cartesian_product_empty() {
        let result = cartesian_product(&[]);
        let expected: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cartesian_product_single() {
        let result = cartesian_product(&[vec![0, 1, 2]]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cartesian_product_two_dims() {
        let result = cartesian_product(&[vec![0, 1], vec![0, 1, 2]]);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_enumerate_indices() {
        let result = enumerate_indices(&[2, 3]);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_template_spec_creation() {
        let base = crate::checker::litmus::mp_test();
        let tmpl = TemplateSpec::new("test-template", base);
        assert_eq!(tmpl.name, "test-template");
        assert!(tmpl.slots.is_empty());
    }

    #[test]
    fn test_mp_template() {
        let tmpl = mp_template();
        assert_eq!(tmpl.slots.len(), 4);
        assert_eq!(tmpl.name, "MP-tmpl");
    }

    #[test]
    fn test_sb_template() {
        let tmpl = sb_template();
        assert_eq!(tmpl.slots.len(), 4);
    }

    #[test]
    fn test_lb_template() {
        let tmpl = lb_template();
        assert_eq!(tmpl.slots.len(), 4);
    }

    #[test]
    fn test_template_instantiation_count() {
        let tmpl = mp_template();
        let config = GeneratorConfig::minimal();
        let count = tmpl.instantiation_count(&config);
        // 2 orderings ^ 4 slots = 16
        assert_eq!(count, 16);
    }

    #[test]
    fn test_template_instantiate_all() {
        let tmpl = mp_template();
        let config = GeneratorConfig::minimal();
        let tests = tmpl.instantiate_all(&config);
        assert_eq!(tests.len(), 16);
        // Each test should have 2 threads
        for t in &tests {
            assert_eq!(t.thread_count(), 2);
        }
    }

    #[test]
    fn test_constraint_min_threads() {
        let test = crate::checker::litmus::mp_test();
        assert!(ConstraintSpec::MinThreads(2).satisfied_by(&test));
        assert!(!ConstraintSpec::MinThreads(3).satisfied_by(&test));
    }

    #[test]
    fn test_constraint_min_locations() {
        let test = crate::checker::litmus::mp_test();
        assert!(ConstraintSpec::MinLocations(2).satisfied_by(&test));
    }

    #[test]
    fn test_constraint_requires_communication() {
        let test = crate::checker::litmus::mp_test();
        assert!(ConstraintSpec::RequiresCommunication.satisfied_by(&test));
    }

    #[test]
    fn test_constraint_requires_fence() {
        let test = crate::checker::litmus::mp_test();
        assert!(!ConstraintSpec::RequiresFence.satisfied_by(&test));
    }

    #[test]
    fn test_pattern_message_passing() {
        let test = crate::checker::litmus::mp_test();
        assert!(check_pattern(&test, &PatternConstraint::MessagePassing));
    }

    #[test]
    fn test_pattern_store_buffering() {
        let test = crate::checker::litmus::sb_test();
        assert!(check_pattern(&test, &PatternConstraint::StoreBuffering));
    }

    #[test]
    fn test_pattern_load_buffering() {
        let test = crate::checker::litmus::lb_test();
        assert!(check_pattern(&test, &PatternConstraint::LoadBuffering));
    }

    #[test]
    fn test_pattern_iriw() {
        let test = crate::checker::litmus::iriw_test();
        assert!(check_pattern(&test, &PatternConstraint::IRIW));
    }

    #[test]
    fn test_family_mp() {
        let family = mp_family();
        assert_eq!(family.name, "MP");
        assert_eq!(family.axes.len(), 2);
        assert_eq!(family.size(), 9); // 3 * 3
    }

    #[test]
    fn test_family_sb() {
        let family = sb_family();
        assert_eq!(family.name, "SB");
        assert_eq!(family.size(), 2);
    }

    #[test]
    fn test_family_lb() {
        let family = lb_family();
        assert_eq!(family.name, "LB");
        assert_eq!(family.size(), 4);
    }

    #[test]
    fn test_family_generate_all() {
        let family = mp_family();
        let tests = family.generate_all();
        assert_eq!(tests.len(), 9);
    }

    #[test]
    fn test_generator_systematic() {
        let config = GeneratorConfig {
            max_threads: 2,
            max_instrs_per_thread: 1,
            max_locations: 1,
            max_value: 1,
            orderings: vec![Ordering::Relaxed],
            include_fences: false,
            include_rmw: false,
            max_tests: 50,
            strategy: GenerationStrategy::Systematic,
            ..Default::default()
        };
        let mut gen = TestGenerator::new(config);
        let count = gen.generate();
        assert!(count > 0);
        assert!(gen.tests().len() <= 50);
    }

    #[test]
    fn test_generator_random() {
        let config = GeneratorConfig {
            max_tests: 10,
            strategy: GenerationStrategy::Random,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        let count = gen.generate();
        assert_eq!(count, 10);
        assert_eq!(gen.tests().len(), 10);
    }

    #[test]
    fn test_generator_with_constraints() {
        let config = GeneratorConfig {
            max_tests: 20,
            strategy: GenerationStrategy::Random,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.add_constraint(ConstraintSpec::RequiresCommunication);
        let count = gen.generate();
        // All generated tests should have shared locations.
        for test in gen.tests() {
            assert!(ConstraintSpec::RequiresCommunication.satisfied_by(test));
        }
        assert!(count > 0);
    }

    #[test]
    fn test_generator_template_based() {
        let config = GeneratorConfig {
            max_tests: 100,
            strategy: GenerationStrategy::Template,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.add_template(mp_template());
        let count = gen.generate();
        assert!(count > 0);
    }

    #[test]
    fn test_generator_into_tests() {
        let config = GeneratorConfig {
            max_tests: 5,
            strategy: GenerationStrategy::Random,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.generate();
        let tests = gen.into_tests();
        assert_eq!(tests.len(), 5);
    }

    #[test]
    fn test_compute_signature() {
        let test = crate::checker::litmus::mp_test();
        let sig = compute_signature(&test);
        assert_eq!(sig.n_threads, 2);
        assert_eq!(sig.n_stores, 2);
        assert_eq!(sig.n_loads, 2);
        assert_eq!(sig.n_fences, 0);
        assert!(sig.has_shared_location);
    }

    #[test]
    fn test_select_diverse() {
        let config = GeneratorConfig {
            max_tests: 50,
            strategy: GenerationStrategy::Random,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.generate();
        let tests = gen.into_tests();
        let selected = select_diverse(tests, 10);
        assert!(selected.len() <= 10);
    }

    #[test]
    fn test_apply_ordering() {
        let mut instr = Instruction::Load { reg: 0, addr: 0x100, ordering: Ordering::Relaxed };
        apply_ordering(&mut instr, Ordering::SeqCst);
        if let Instruction::Load { ordering, .. } = instr {
            assert_eq!(ordering, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_apply_address() {
        let mut instr = Instruction::Store { addr: 0x100, value: 1, ordering: Ordering::Relaxed };
        apply_address(&mut instr, 0x200);
        if let Instruction::Store { addr, .. } = instr {
            assert_eq!(addr, 0x200);
        }
    }

    #[test]
    fn test_apply_value() {
        let mut instr = Instruction::Store { addr: 0x100, value: 1, ordering: Ordering::Relaxed };
        apply_value(&mut instr, 42);
        if let Instruction::Store { value, .. } = instr {
            assert_eq!(value, 42);
        }
    }

    #[test]
    fn test_apply_scope() {
        let mut instr = Instruction::Fence { ordering: Ordering::SeqCst, scope: Scope::None };
        apply_scope(&mut instr, Scope::CTA);
        if let Instruction::Fence { scope, .. } = instr {
            assert_eq!(scope, Scope::CTA);
        }
    }

    #[test]
    fn test_wrc_variants() {
        let config = GeneratorConfig::minimal();
        let tests = wrc_variants(&config);
        assert!(!tests.is_empty());
        for t in &tests {
            assert_eq!(t.thread_count(), 3);
        }
    }

    #[test]
    fn test_isa2_variants() {
        let config = GeneratorConfig::minimal();
        let tests = isa2_variants(&config);
        assert!(!tests.is_empty());
        for t in &tests {
            assert_eq!(t.thread_count(), 3);
        }
    }

    #[test]
    fn test_r_pattern_variants() {
        let config = GeneratorConfig::minimal();
        let tests = r_pattern_variants(&config);
        assert!(!tests.is_empty());
        for t in &tests {
            assert_eq!(t.thread_count(), 2);
        }
    }

    #[test]
    fn test_s_pattern_variants() {
        let config = GeneratorConfig::minimal();
        let tests = s_pattern_variants(&config);
        assert!(!tests.is_empty());
        for t in &tests {
            assert_eq!(t.thread_count(), 3);
        }
    }

    #[test]
    fn test_rmw_variants() {
        let config = GeneratorConfig::minimal();
        let tests = rmw_variants(&config);
        assert!(!tests.is_empty());
    }

    #[test]
    fn test_fence_variants() {
        let base = crate::checker::litmus::mp_test();
        let config = GeneratorConfig::minimal();
        let tests = fence_variants(&base, &config);
        assert!(!tests.is_empty());
        // Each variant should have more instructions than the base.
        for t in &tests {
            assert!(t.total_instructions() > base.total_instructions());
        }
    }

    #[test]
    fn test_gpu_mp_variants() {
        let config = GeneratorConfig::gpu();
        let tests = gpu_mp_variants(&config);
        assert_eq!(tests.len(), 9); // 3 scopes × 3 scopes
    }

    #[test]
    fn test_gpu_sb_variants() {
        let config = GeneratorConfig::gpu();
        let tests = gpu_sb_variants(&config);
        assert_eq!(tests.len(), 3); // 3 scopes
    }

    #[test]
    fn test_comprehensive_suite() {
        let config = GeneratorConfig {
            max_tests: 50,
            ..GeneratorConfig::minimal()
        };
        let tests = generate_comprehensive_suite(&config);
        assert!(!tests.is_empty());
        assert!(tests.len() <= 50);
    }

    #[test]
    fn test_scope_to_release() {
        assert_eq!(scope_to_release(Scope::CTA), Ordering::ReleaseCTA);
        assert_eq!(scope_to_release(Scope::GPU), Ordering::ReleaseGPU);
        assert_eq!(scope_to_release(Scope::System), Ordering::ReleaseSystem);
        assert_eq!(scope_to_release(Scope::None), Ordering::Release);
    }

    #[test]
    fn test_scope_to_acquire() {
        assert_eq!(scope_to_acquire(Scope::CTA), Ordering::AcquireCTA);
        assert_eq!(scope_to_acquire(Scope::GPU), Ordering::AcquireGPU);
        assert_eq!(scope_to_acquire(Scope::System), Ordering::AcquireSystem);
        assert_eq!(scope_to_acquire(Scope::None), Ordering::Acquire);
    }

    #[test]
    fn test_generation_strategy_display() {
        assert_eq!(format!("{}", GenerationStrategy::Systematic), "systematic");
        assert_eq!(format!("{}", GenerationStrategy::Template), "template");
        assert_eq!(format!("{}", GenerationStrategy::Random), "random");
    }

    #[test]
    fn test_slot_kind_equality() {
        assert_eq!(SlotKind::Ordering, SlotKind::Ordering);
        assert_ne!(SlotKind::Ordering, SlotKind::Address);
    }

    #[test]
    fn test_generator_default() {
        let gen = TestGenerator::default_generator();
        assert!(gen.tests().is_empty());
    }

    #[test]
    fn test_two_plus_two_w_template() {
        let tmpl = two_plus_two_w_template();
        assert_eq!(tmpl.slots.len(), 4);
    }

    #[test]
    fn test_constraint_custom() {
        let test = crate::checker::litmus::mp_test();
        let c = ConstraintSpec::Custom("test".into());
        assert!(c.satisfied_by(&test));
    }

    #[test]
    fn test_family_empty_axes() {
        let base = crate::checker::litmus::mp_test();
        let family = TestFamily::new("empty", base);
        assert_eq!(family.size(), 1);
        let tests = family.generate_all();
        assert_eq!(tests.len(), 1);
    }

    #[test]
    fn test_coverage_guided_generation() {
        let config = GeneratorConfig {
            max_tests: 10,
            strategy: GenerationStrategy::CoverageGuided,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        let count = gen.generate();
        assert!(count > 0);
    }

    #[test]
    fn test_constrained_generation() {
        let config = GeneratorConfig {
            max_tests: 10,
            strategy: GenerationStrategy::Constraint,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.add_constraint(ConstraintSpec::MinThreads(2));
        let count = gen.generate();
        assert!(count > 0);
        for test in gen.tests() {
            assert!(test.thread_count() >= 2);
        }
    }

    #[test]
    fn test_multiple_constraints() {
        let config = GeneratorConfig {
            max_tests: 10,
            strategy: GenerationStrategy::Random,
            ..GeneratorConfig::minimal()
        };
        let mut gen = TestGenerator::new(config);
        gen.add_constraint(ConstraintSpec::MinThreads(2));
        gen.add_constraint(ConstraintSpec::RequiresCommunication);
        let count = gen.generate();
        for test in gen.tests() {
            assert!(test.thread_count() >= 2);
            assert!(ConstraintSpec::RequiresCommunication.satisfied_by(test));
        }
        assert!(count > 0);
    }
}
