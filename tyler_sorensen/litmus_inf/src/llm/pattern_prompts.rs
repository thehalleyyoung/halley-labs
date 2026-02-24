//! Pattern-specific prompt templates for litmus test generation.
//!
//! Curated prompts organised by bug class, memory model, and concurrency
//! pattern.  Each entry includes a system instruction, 3-5 examples with
//! expected output, constraints, and output-format specification.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::checker::litmus::Ordering;

use super::prompt_engine::{ShotExample, OutputFormat};

// ---------------------------------------------------------------------------
// Core library
// ---------------------------------------------------------------------------

/// Collection of curated prompts indexed by pattern name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPromptLibrary {
    /// Bug-class prompts (data race, use-after-free, etc.).
    pub bug_prompts: Vec<BugPatternPrompt>,
    /// Memory-model-specific prompts (TSO, ARM, GPU, C++11).
    pub model_prompts: Vec<ModelSpecificPrompt>,
    /// Generic pattern prompts (SB, MP, LB, IRIW, …).
    pub pattern_prompts: HashMap<String, PatternPromptEntry>,
}

/// A single prompt entry for a named pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPromptEntry {
    pub name: String,
    pub system_instruction: String,
    pub examples: Vec<ShotExample>,
    pub constraints: Vec<String>,
    pub output_format: OutputFormat,
    pub description: String,
}

impl Default for PatternPromptLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternPromptLibrary {
    /// Build the full library with all built-in prompts.
    pub fn new() -> Self {
        let mut lib = Self {
            bug_prompts: Vec::new(),
            model_prompts: Vec::new(),
            pattern_prompts: HashMap::new(),
        };
        lib.register_bug_prompts();
        lib.register_model_prompts();
        lib.register_generic_patterns();
        lib
    }

    /// Look up a prompt entry by pattern name (case-insensitive).
    pub fn get_pattern(&self, name: &str) -> Option<&PatternPromptEntry> {
        self.pattern_prompts.get(&name.to_uppercase())
    }

    /// Look up bug-class prompts matching a tag.
    pub fn get_bug_prompts(&self, bug_class: &str) -> Vec<&BugPatternPrompt> {
        let lc = bug_class.to_lowercase();
        self.bug_prompts
            .iter()
            .filter(|p| p.bug_class.to_lowercase().contains(&lc))
            .collect()
    }

    /// Look up model-specific prompts.
    pub fn get_model_prompts(&self, model: &str) -> Vec<&ModelSpecificPrompt> {
        let lc = model.to_lowercase();
        self.model_prompts
            .iter()
            .filter(|p| p.model_family.to_lowercase().contains(&lc))
            .collect()
    }

    /// All pattern names.
    pub fn pattern_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.pattern_prompts.keys().cloned().collect();
        names.sort();
        names
    }

    /// All bug classes.
    pub fn bug_classes(&self) -> Vec<String> {
        let mut classes: Vec<_> = self
            .bug_prompts
            .iter()
            .map(|p| p.bug_class.clone())
            .collect();
        classes.sort();
        classes.dedup();
        classes
    }

    // -----------------------------------------------------------------------
    // Generic patterns
    // -----------------------------------------------------------------------

    fn register_generic_patterns(&mut self) {
        self.register_sb();
        self.register_mp();
        self.register_lb();
        self.register_iriw();
        self.register_two_plus_two_w();
        self.register_rwc();
        self.register_wrc();
    }

    fn register_sb(&mut self) {
        let entry = PatternPromptEntry {
            name: "SB".to_string(),
            description: "Store-Buffering: two threads each write then read the other's location"
                .to_string(),
            system_instruction: "\
You are generating a Store-Buffering (SB) litmus test.
SB tests whether a processor can observe its own store before it becomes visible to other threads.
The classic shape: Thread 0 writes x then reads y; Thread 1 writes y then reads x.
The weak outcome (both reads see 0) is allowed on TSO and weaker models."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic SB with relaxed ordering",
                    "Generate SB test with relaxed memory ordering, 2 threads",
                    "test SB\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  load r0 y relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Allowed",
                    "SB",
                ),
                ShotExample::new(
                    "SB with fences (forbidden outcome under SC)",
                    "Generate SB test with seq_cst fences between store and load",
                    "test SB_fenced\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence seq_cst\n  load r0 y relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  fence seq_cst\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Forbidden",
                    "SB",
                ),
                ShotExample::new(
                    "SB with 3 variables",
                    "Generate SB variant with 3 shared variables",
                    "test SB3\ninit x = 0\ninit y = 0\ninit z = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  load r0 y relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  load r0 z relaxed\n\n\
                     Thread 2:\n  store z 1 relaxed\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0, 2:r0=0 -> Allowed",
                    "SB",
                ),
            ],
            constraints: vec![
                "Each thread must have at least one store and one load".to_string(),
                "Stores and loads must access different variables within a thread".to_string(),
                "All variables must be initialised to 0".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("SB".to_string(), entry);
    }

    fn register_mp(&mut self) {
        let entry = PatternPromptEntry {
            name: "MP".to_string(),
            description: "Message-Passing: writer publishes data then flag; reader checks flag then data"
                .to_string(),
            system_instruction: "\
You are generating a Message-Passing (MP) litmus test.
MP tests whether a store to a data location followed by a store to a flag
is observed in order by another thread that reads the flag then the data.
With release/acquire ordering, the weak outcome (flag seen but data stale)
should be forbidden."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic MP with release/acquire",
                    "Generate MP test with release on the flag store, acquire on flag load",
                    "test MP\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release\n\n\
                     Thread 1:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "MP",
                ),
                ShotExample::new(
                    "MP with relaxed ordering (weak outcome allowed)",
                    "Generate MP test with all relaxed ordering",
                    "test MP_relaxed\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "MP",
                ),
                ShotExample::new(
                    "MP with fence instead of release/acquire",
                    "Generate MP test using fences for ordering",
                    "test MP_fence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence release\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  fence acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "MP",
                ),
                ShotExample::new(
                    "MP with multiple data variables",
                    "Generate MP test publishing two data values behind a flag",
                    "test MP_multi\ninit x = 0\ninit y = 0\ninit z = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 2 relaxed\n  store z 1 release\n\n\
                     Thread 1:\n  load r0 z acquire\n  load r1 x relaxed\n  load r2 y relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "MP",
                ),
            ],
            constraints: vec![
                "Thread 0 must store data before the flag".to_string(),
                "Thread 1 must load the flag before the data".to_string(),
                "The flag variable must be different from the data variable(s)".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("MP".to_string(), entry);
    }

    fn register_lb(&mut self) {
        let entry = PatternPromptEntry {
            name: "LB".to_string(),
            description: "Load-Buffering: two threads read then write different locations"
                .to_string(),
            system_instruction: "\
You are generating a Load-Buffering (LB) litmus test.
LB tests whether a load can be reordered after a subsequent store on the same thread.
The weak outcome (both loads see the other thread's store) is forbidden on TSO
but allowed on ARM/RISC-V."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic LB with relaxed ordering",
                    "Generate LB test with relaxed memory ordering",
                    "test LB\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  load r0 x relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  store x 1 relaxed\n\n\
                     outcome: 0:r0=1, 1:r0=1 -> Allowed",
                    "LB",
                ),
                ShotExample::new(
                    "LB with acquire loads (forbidden on most models)",
                    "Generate LB test with acquire ordering on loads",
                    "test LB_acq\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  load r0 x acquire\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y acquire\n  store x 1 relaxed\n\n\
                     outcome: 0:r0=1, 1:r0=1 -> Forbidden",
                    "LB",
                ),
                ShotExample::new(
                    "LB with dependencies",
                    "Generate LB test with data dependency from load to store",
                    "test LB_dep\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  load r0 x relaxed\n  store y r0 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  store x 1 relaxed\n\n\
                     outcome: 0:r0=1, 1:r0=1 -> Forbidden",
                    "LB",
                ),
            ],
            constraints: vec![
                "Each thread must load before storing".to_string(),
                "Threads must access different variables for load and store".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("LB".to_string(), entry);
    }

    fn register_iriw(&mut self) {
        let entry = PatternPromptEntry {
            name: "IRIW".to_string(),
            description: "Independent-Reads-Independent-Writes: 4 threads test multi-copy atomicity"
                .to_string(),
            system_instruction: "\
You are generating an IRIW (Independent Reads of Independent Writes) litmus test.
IRIW tests multi-copy atomicity: whether two readers can observe two independent
writes in different orders. This requires 4 threads: two writers and two readers.
The weak outcome is forbidden under multi-copy-atomic models but allowed under
non-MCA models."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic IRIW",
                    "Generate IRIW test with 4 threads",
                    "test IRIW\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 x acquire\n  load r1 y acquire\n\n\
                     Thread 3:\n  load r0 y acquire\n  load r1 x acquire\n\n\
                     outcome: 2:r0=1, 2:r1=0, 3:r0=1, 3:r1=0 -> Forbidden",
                    "IRIW",
                ),
                ShotExample::new(
                    "IRIW with seq_cst fences",
                    "Generate IRIW test with fences on reader threads",
                    "test IRIW_fence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 x relaxed\n  fence seq_cst\n  load r1 y relaxed\n\n\
                     Thread 3:\n  load r0 y relaxed\n  fence seq_cst\n  load r1 x relaxed\n\n\
                     outcome: 2:r0=1, 2:r1=0, 3:r0=1, 3:r1=0 -> Forbidden",
                    "IRIW",
                ),
                ShotExample::new(
                    "IRIW with relaxed (weak outcome allowed)",
                    "Generate IRIW test where weak outcome is allowed",
                    "test IRIW_relaxed\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 x relaxed\n  load r1 y relaxed\n\n\
                     Thread 3:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 2:r0=1, 2:r1=0, 3:r0=1, 3:r1=0 -> Allowed",
                    "IRIW",
                ),
            ],
            constraints: vec![
                "Must have exactly 4 threads".to_string(),
                "Threads 0 and 1 are writers; threads 2 and 3 are readers".to_string(),
                "Each writer writes to a different variable".to_string(),
                "Each reader reads both variables".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("IRIW".to_string(), entry);
    }

    fn register_two_plus_two_w(&mut self) {
        let entry = PatternPromptEntry {
            name: "2+2W".to_string(),
            description: "2+2W coherence: two threads each write two locations in opposite order"
                .to_string(),
            system_instruction: "\
You are generating a 2+2W litmus test.
This is a coherence test: two threads each write to two shared variables
but in opposite order. The test checks whether the stores can interleave
in an unexpected way."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic 2+2W",
                    "Generate 2+2W test",
                    "test 2+2W\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 2 relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  store x 2 relaxed\n\n\
                     outcome: x=1, y=1 -> Allowed",
                    "2+2W",
                ),
                ShotExample::new(
                    "2+2W with fences",
                    "Generate 2+2W with fences to forbid interleaving",
                    "test 2+2W_fenced\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence seq_cst\n  store y 2 relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  fence seq_cst\n  store x 2 relaxed\n\n\
                     outcome: x=1, y=1 -> Forbidden",
                    "2+2W",
                ),
                ShotExample::new(
                    "2+2W with release stores",
                    "Generate 2+2W with release ordering",
                    "test 2+2W_rel\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 release\n  store y 2 release\n\n\
                     Thread 1:\n  store y 1 release\n  store x 2 release\n\n\
                     outcome: x=1, y=1 -> Allowed",
                    "2+2W",
                ),
            ],
            constraints: vec![
                "Each thread writes to both shared variables".to_string(),
                "The write order must be opposite between threads".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("2+2W".to_string(), entry);
    }

    fn register_rwc(&mut self) {
        let entry = PatternPromptEntry {
            name: "RWC".to_string(),
            description: "Read-Write-Coherence: 3 threads testing store order observation"
                .to_string(),
            system_instruction: "\
You are generating an RWC (Read-Write-Coherence) litmus test.
RWC uses 3 threads: Thread 0 writes x; Thread 1 reads x then writes y;
Thread 2 reads y then reads x. It tests whether coherence-ordered writes
are seen consistently."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic RWC",
                    "Generate RWC test with 3 threads",
                    "test RWC\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x acquire\n  store y 1 release\n\n\
                     Thread 2:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Forbidden",
                    "RWC",
                ),
                ShotExample::new(
                    "RWC with relaxed ordering",
                    "Generate RWC test where weak outcome is allowed",
                    "test RWC_relaxed\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Allowed",
                    "RWC",
                ),
                ShotExample::new(
                    "RWC with fences",
                    "Generate RWC test with fences",
                    "test RWC_fence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n  fence seq_cst\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 y relaxed\n  fence seq_cst\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Forbidden",
                    "RWC",
                ),
            ],
            constraints: vec![
                "Must have 3 threads".to_string(),
                "Thread 0 writes; Thread 1 reads then writes; Thread 2 reads twice".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("RWC".to_string(), entry);
    }

    fn register_wrc(&mut self) {
        let entry = PatternPromptEntry {
            name: "WRC".to_string(),
            description: "Write-Read-Coherence: 3-thread message-passing chain".to_string(),
            system_instruction: "\
You are generating a WRC (Write-Read-Coherence) litmus test.
WRC is a 3-thread chain: Thread 0 writes x; Thread 1 reads x then writes y;
Thread 2 reads y then reads x. Tests whether the write to x propagates
through the chain."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic WRC",
                    "Generate WRC test",
                    "test WRC\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 release\n\n\
                     Thread 1:\n  load r0 x acquire\n  store y 1 release\n\n\
                     Thread 2:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Forbidden",
                    "WRC",
                ),
                ShotExample::new(
                    "WRC with relaxed",
                    "Generate WRC test where weak outcome is allowed",
                    "test WRC_relaxed\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Allowed",
                    "WRC",
                ),
                ShotExample::new(
                    "WRC with seq_cst fences",
                    "Generate WRC test with fences",
                    "test WRC_fence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n  fence seq_cst\n  store y 1 relaxed\n\n\
                     Thread 2:\n  load r0 y relaxed\n  fence seq_cst\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 2:r0=1, 2:r1=0 -> Forbidden",
                    "WRC",
                ),
            ],
            constraints: vec![
                "Must have 3 threads forming a chain".to_string(),
                "Thread 0 → Thread 1 → Thread 2 through shared variables".to_string(),
            ],
            output_format: OutputFormat::Pseudocode,
        };
        self.pattern_prompts.insert("WRC".to_string(), entry);
    }

    // -----------------------------------------------------------------------
    // Bug-class prompts
    // -----------------------------------------------------------------------

    fn register_bug_prompts(&mut self) {
        self.bug_prompts.push(Self::data_race_prompt());
        self.bug_prompts.push(Self::use_after_free_prompt());
        self.bug_prompts.push(Self::aba_problem_prompt());
        self.bug_prompts.push(Self::ordering_violation_prompt());
        self.bug_prompts.push(Self::publication_failure_prompt());
    }

    fn data_race_prompt() -> BugPatternPrompt {
        BugPatternPrompt {
            bug_class: "data-race".to_string(),
            description: "Tests for unprotected concurrent access to shared data".to_string(),
            system_instruction: "\
You are generating litmus tests that expose data races.
A data race occurs when two threads access the same memory location
without synchronisation, and at least one access is a write.
Generate tests where the weak outcome demonstrates the data race."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Simple data race on x",
                    "Generate a test exposing a data race on variable x",
                    "test DataRace\ninit x = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n\n\
                     outcome: 1:r0=0 -> Allowed",
                    "data-race",
                ),
                ShotExample::new(
                    "Data race with two writers",
                    "Generate a test with two threads writing to the same location",
                    "test WriteWriteRace\ninit x = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n\n\
                     Thread 1:\n  store x 2 relaxed\n\n\
                     outcome: x=1 -> Allowed",
                    "data-race",
                ),
                ShotExample::new(
                    "Data race with read-modify-write",
                    "Generate a test exposing a race between a plain load and an RMW",
                    "test RMWrace\ninit x = 0\n\n\
                     Thread 0:\n  rmw r0 x 1 relaxed\n\n\
                     Thread 1:\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Allowed",
                    "data-race",
                ),
            ],
            required_orderings: vec![Ordering::Relaxed],
            typical_thread_count: 2,
        }
    }

    fn use_after_free_prompt() -> BugPatternPrompt {
        BugPatternPrompt {
            bug_class: "use-after-free".to_string(),
            description: "Tests modelling concurrent allocation/deallocation patterns".to_string(),
            system_instruction: "\
You are generating litmus tests that model use-after-free scenarios.
In concurrent code, one thread may free a resource while another still
holds a reference. Model this with a flag variable indicating 'freed'
and data accesses that should not occur after the flag is set."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Use-after-free: reader vs. freer",
                    "Generate a test where thread 1 reads data after thread 0 'frees' it",
                    "test UAF\ninit x = 42\ninit freed = 0\n\n\
                     Thread 0:\n  store freed 1 release\n  store x 0 relaxed\n\n\
                     Thread 1:\n  load r0 freed acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=42 -> Allowed",
                    "use-after-free",
                ),
                ShotExample::new(
                    "Double-free modelling",
                    "Generate a test where two threads both attempt to free",
                    "test DoubleFree\ninit freed = 0\n\n\
                     Thread 0:\n  rmw r0 freed 1 acq_rel\n\n\
                     Thread 1:\n  rmw r0 freed 1 acq_rel\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Forbidden",
                    "use-after-free",
                ),
                ShotExample::new(
                    "Lazy reclamation race",
                    "Generate a test modelling hazard-pointer-like reclamation",
                    "test LazyReclaim\ninit x = 1\ninit guard = 0\ninit freed = 0\n\n\
                     Thread 0:\n  store guard 1 release\n  load r0 x relaxed\n\n\
                     Thread 1:\n  load r0 guard acquire\n  store freed 1 release\n  store x 0 relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=1 -> Allowed",
                    "use-after-free",
                ),
            ],
            required_orderings: vec![Ordering::Acquire, Ordering::Release],
            typical_thread_count: 2,
        }
    }

    fn aba_problem_prompt() -> BugPatternPrompt {
        BugPatternPrompt {
            bug_class: "aba-problem".to_string(),
            description: "Tests for ABA problems in lock-free data structures".to_string(),
            system_instruction: "\
You are generating litmus tests that model ABA problems.
An ABA problem occurs when a CAS operation succeeds because the target
has the expected value, but the value has changed and changed back in
between. Thread 0 reads A, then Thread 1 changes A→B→A, then Thread 0's
CAS succeeds incorrectly."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Classic ABA",
                    "Generate a test modelling the ABA problem with CAS",
                    "test ABA\ninit x = 1\n\n\
                     Thread 0:\n  load r0 x relaxed\n  rmw r1 x 3 acq_rel\n\n\
                     Thread 1:\n  rmw r0 x 2 acq_rel\n  rmw r1 x 1 acq_rel\n\n\
                     outcome: 0:r0=1, 0:r1=1 -> Allowed",
                    "aba-problem",
                ),
                ShotExample::new(
                    "ABA with counter",
                    "Generate an ABA test using a version counter to detect the problem",
                    "test ABA_counter\ninit x = 1\ninit ver = 0\n\n\
                     Thread 0:\n  load r0 x relaxed\n  load r1 ver relaxed\n\n\
                     Thread 1:\n  store x 2 relaxed\n  store ver 1 release\n  store x 1 relaxed\n  store ver 2 release\n\n\
                     outcome: 0:r0=1, 0:r1=0 -> Allowed",
                    "aba-problem",
                ),
                ShotExample::new(
                    "ABA with three threads",
                    "Generate an ABA scenario with a third thread observing",
                    "test ABA3\ninit x = 1\n\n\
                     Thread 0:\n  load r0 x acquire\n\n\
                     Thread 1:\n  rmw r0 x 2 acq_rel\n\n\
                     Thread 2:\n  rmw r0 x 1 acq_rel\n\n\
                     outcome: 0:r0=1, 1:r0=1, 2:r0=2 -> Allowed",
                    "aba-problem",
                ),
            ],
            required_orderings: vec![Ordering::AcqRel],
            typical_thread_count: 2,
        }
    }

    fn ordering_violation_prompt() -> BugPatternPrompt {
        BugPatternPrompt {
            bug_class: "ordering-violation".to_string(),
            description: "Tests for incorrect assumptions about memory ordering".to_string(),
            system_instruction: "\
You are generating litmus tests that expose ordering violations.
An ordering violation occurs when a programmer assumes sequential consistency
but uses insufficiently strong memory orderings. The test should show an
outcome that is forbidden under SC but allowed under the weaker model."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Missing acquire on flag check",
                    "Generate a test where a relaxed flag check leads to a violation",
                    "test OrdViolation\ninit x = 0\ninit flag = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store flag 1 release\n\n\
                     Thread 1:\n  load r0 flag relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "ordering-violation",
                ),
                ShotExample::new(
                    "Store-store reordering",
                    "Generate a test exposing store-store reordering",
                    "test StoreReorder\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "ordering-violation",
                ),
                ShotExample::new(
                    "Load-load reordering",
                    "Generate a test exposing load-load reordering",
                    "test LoadReorder\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 release\n  store y 1 release\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "ordering-violation",
                ),
                ShotExample::new(
                    "Missing fence",
                    "Generate a test showing a bug due to missing fence",
                    "test MissingFence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "ordering-violation",
                ),
            ],
            required_orderings: vec![Ordering::Relaxed],
            typical_thread_count: 2,
        }
    }

    fn publication_failure_prompt() -> BugPatternPrompt {
        BugPatternPrompt {
            bug_class: "publication-failure".to_string(),
            description: "Tests for safe publication violations".to_string(),
            system_instruction: "\
You are generating litmus tests that expose publication failures.
A publication failure occurs when an object is constructed and then
a pointer/flag is published, but without proper release/acquire
synchronisation, so another thread may see the published pointer
but read uninitialised data."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "Publication without release",
                    "Generate a test where data is published without release ordering",
                    "test PubFail\ninit data = 0\ninit ready = 0\n\n\
                     Thread 0:\n  store data 42 relaxed\n  store ready 1 relaxed\n\n\
                     Thread 1:\n  load r0 ready relaxed\n  load r1 data relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "publication-failure",
                ),
                ShotExample::new(
                    "Correct publication with release/acquire",
                    "Generate a test where data is properly published",
                    "test PubOK\ninit data = 0\ninit ready = 0\n\n\
                     Thread 0:\n  store data 42 relaxed\n  store ready 1 release\n\n\
                     Thread 1:\n  load r0 ready acquire\n  load r1 data relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "publication-failure",
                ),
                ShotExample::new(
                    "Multi-field publication failure",
                    "Generate a test with two data fields published behind a flag",
                    "test PubFail2\ninit a = 0\ninit b = 0\ninit ready = 0\n\n\
                     Thread 0:\n  store a 1 relaxed\n  store b 2 relaxed\n  store ready 1 relaxed\n\n\
                     Thread 1:\n  load r0 ready relaxed\n  load r1 a relaxed\n  load r2 b relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "publication-failure",
                ),
                ShotExample::new(
                    "Publication with fence",
                    "Generate a test using a fence for safe publication",
                    "test PubFence\ninit data = 0\ninit ready = 0\n\n\
                     Thread 0:\n  store data 42 relaxed\n  fence release\n  store ready 1 relaxed\n\n\
                     Thread 1:\n  load r0 ready relaxed\n  fence acquire\n  load r1 data relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "publication-failure",
                ),
            ],
            required_orderings: vec![Ordering::Relaxed, Ordering::Release, Ordering::Acquire],
            typical_thread_count: 2,
        }
    }

    // -----------------------------------------------------------------------
    // Model-specific prompts
    // -----------------------------------------------------------------------

    fn register_model_prompts(&mut self) {
        self.model_prompts.push(Self::tso_prompt());
        self.model_prompts.push(Self::arm_riscv_prompt());
        self.model_prompts.push(Self::gpu_prompt());
        self.model_prompts.push(Self::cpp11_prompt());
    }

    fn tso_prompt() -> ModelSpecificPrompt {
        ModelSpecificPrompt {
            model_family: "TSO".to_string(),
            description: "x86-TSO / SPARC-TSO store buffer tests".to_string(),
            system_instruction: "\
You are generating litmus tests for TSO (Total Store Order), the memory model
used by x86 and SPARC processors.
Under TSO, the only reordering allowed is store→load: a store can be buffered
and a subsequent load (to a different address) can overtake it. All other
orderings are preserved. Store-Buffering (SB) is the canonical TSO test."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "TSO store buffer (SB allowed)",
                    "Generate a TSO store-buffer test",
                    "test TSO_SB\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  load r0 y relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Allowed",
                    "TSO-SB",
                ),
                ShotExample::new(
                    "TSO forbids load-load reordering",
                    "Generate a test showing TSO preserves load-load order",
                    "test TSO_NoLLReorder\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "TSO-LL",
                ),
                ShotExample::new(
                    "TSO forbids store-store reordering",
                    "Generate a test showing TSO preserves store-store order",
                    "test TSO_NoSSReorder\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "TSO-SS",
                ),
                ShotExample::new(
                    "TSO with mfence eliminates SB",
                    "Generate a test showing mfence prevents store buffering on TSO",
                    "test TSO_MFENCE\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence seq_cst\n  load r0 y relaxed\n\n\
                     Thread 1:\n  store y 1 relaxed\n  fence seq_cst\n  load r0 x relaxed\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Forbidden",
                    "TSO-SB-FENCE",
                ),
            ],
            key_properties: vec![
                "Store-load reordering is allowed (store buffering)".to_string(),
                "Load-load order is preserved".to_string(),
                "Store-store order is preserved".to_string(),
                "Load-store order is preserved".to_string(),
                "MFENCE drains the store buffer".to_string(),
            ],
        }
    }

    fn arm_riscv_prompt() -> ModelSpecificPrompt {
        ModelSpecificPrompt {
            model_family: "ARM/RISC-V".to_string(),
            description: "ARM and RISC-V multi-copy atomicity tests".to_string(),
            system_instruction: "\
You are generating litmus tests for ARM/RISC-V memory models.
These are weaker than TSO: they allow load-load, load-store, and store-store
reordering in addition to store-load reordering. They are multi-copy atomic
(ARMv8) or have RVWMO (RISC-V). Dependencies (address, data, control) provide
ordering guarantees."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "ARM MP with DMB",
                    "Generate an ARM message-passing test with DMB barriers",
                    "test ARM_MP\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence release\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  fence acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "ARM-MP",
                ),
                ShotExample::new(
                    "ARM LB (load-buffering allowed)",
                    "Generate an ARM load-buffering test",
                    "test ARM_LB\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  load r0 x relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  store x 1 relaxed\n\n\
                     outcome: 0:r0=1, 1:r0=1 -> Allowed",
                    "ARM-LB",
                ),
                ShotExample::new(
                    "ARM store-store reordering",
                    "Generate an ARM test showing store-store reordering",
                    "test ARM_SS\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "ARM-SS",
                ),
                ShotExample::new(
                    "ARM acquire/release pair",
                    "Generate an ARM test using LDAR/STLR",
                    "test ARM_AcqRel\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release\n\n\
                     Thread 1:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "ARM-AcqRel",
                ),
                ShotExample::new(
                    "RISC-V RVWMO: data dependency preserves order",
                    "Generate a RISC-V test showing data dependency ordering",
                    "test RISCV_dep\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release\n\n\
                     Thread 1:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "RISCV-dep",
                ),
            ],
            key_properties: vec![
                "All four reorderings possible without barriers".to_string(),
                "Address/data/control dependencies provide ordering".to_string(),
                "DMB (ARM) / fence (RISC-V) for explicit ordering".to_string(),
                "LDAR/STLR (ARM) for acquire/release".to_string(),
                "Multi-copy atomic (ARMv8): IRIW forbidden with acquires".to_string(),
            ],
        }
    }

    fn gpu_prompt() -> ModelSpecificPrompt {
        ModelSpecificPrompt {
            model_family: "GPU".to_string(),
            description: "GPU memory model tests with scoped synchronisation".to_string(),
            system_instruction: "\
You are generating litmus tests for GPU memory models (CUDA, OpenCL, Vulkan).
GPUs have hierarchical scope: CTA/workgroup, GPU/device, System.
Synchronisation operations have a scope that determines visibility.
A release at CTA scope only synchronises within the CTA; system scope
synchronises across the entire system."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "GPU MP within CTA scope",
                    "Generate a GPU message-passing test with CTA-scope synchronisation",
                    "test GPU_MP_CTA\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release_cta\n\n\
                     Thread 1:\n  load r0 y acquire_cta\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "GPU-MP-CTA",
                ),
                ShotExample::new(
                    "GPU cross-CTA needs system scope",
                    "Generate a GPU test showing CTA scope insufficient across CTAs",
                    "test GPU_CrossCTA\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release_cta\n\n\
                     Thread 1:\n  load r0 y acquire_cta\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "GPU-CrossCTA",
                ),
                ShotExample::new(
                    "GPU system-scope MP",
                    "Generate a GPU test with system scope that works across CTAs",
                    "test GPU_MP_Sys\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release_system\n\n\
                     Thread 1:\n  load r0 y acquire_system\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "GPU-MP-SYS",
                ),
                ShotExample::new(
                    "GPU scoped fence",
                    "Generate a GPU test using a GPU-scope fence",
                    "test GPU_fence\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  fence release gpu\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  fence acquire gpu\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "GPU-fence",
                ),
                ShotExample::new(
                    "GPU scope mismatch",
                    "Generate a test showing mismatched scopes fail to synchronise",
                    "test GPU_ScopeMismatch\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release_cta\n\n\
                     Thread 1:\n  load r0 y acquire_gpu\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "GPU-mismatch",
                ),
            ],
            key_properties: vec![
                "Scoped synchronisation: CTA < GPU < System".to_string(),
                "Release/acquire must have matching or broader scope".to_string(),
                "Threads within the same CTA can use CTA scope".to_string(),
                "Cross-CTA communication requires GPU or System scope".to_string(),
                "CPU-GPU communication requires System scope".to_string(),
            ],
        }
    }

    fn cpp11_prompt() -> ModelSpecificPrompt {
        ModelSpecificPrompt {
            model_family: "C++11".to_string(),
            description: "C++11/C11 memory model tests including release sequences".to_string(),
            system_instruction: "\
You are generating litmus tests for the C++11/C11 memory model.
Key features: relaxed, acquire, release, acq_rel, seq_cst orderings.
Release sequences allow a chain of RMW operations to extend a
release-acquire synchronisation. Consume ordering (deprecated in C++17)
uses dependency-based ordering."
                .to_string(),
            examples: vec![
                ShotExample::new(
                    "C++11 release sequence",
                    "Generate a test showing a release sequence extending through RMW",
                    "test Cpp11_RelSeq\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 release\n\n\
                     Thread 1:\n  rmw r0 y 2 relaxed\n\n\
                     Thread 2:\n  load r0 y acquire\n  load r1 x relaxed\n\n\
                     outcome: 2:r0=2, 2:r1=0 -> Forbidden",
                    "C++11-relseq",
                ),
                ShotExample::new(
                    "C++11 SC atomics total order",
                    "Generate a test showing seq_cst establishes total order",
                    "test Cpp11_SC\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 seq_cst\n\n\
                     Thread 1:\n  store y 1 seq_cst\n\n\
                     Thread 2:\n  load r0 x seq_cst\n  load r1 y seq_cst\n\n\
                     Thread 3:\n  load r0 y seq_cst\n  load r1 x seq_cst\n\n\
                     outcome: 2:r0=1, 2:r1=0, 3:r0=1, 3:r1=0 -> Forbidden",
                    "C++11-SC",
                ),
                ShotExample::new(
                    "C++11 relaxed atomics (no ordering)",
                    "Generate a test with relaxed atomics allowing all interleavings",
                    "test Cpp11_Relaxed\ninit x = 0\ninit y = 0\n\n\
                     Thread 0:\n  store x 1 relaxed\n  store y 1 relaxed\n\n\
                     Thread 1:\n  load r0 y relaxed\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Allowed",
                    "C++11-relaxed",
                ),
                ShotExample::new(
                    "C++11 MP with release/acquire",
                    "Generate a standard message-passing test with C++11 orderings",
                    "test Cpp11_MP\ninit x = 0\ninit flag = 0\n\n\
                     Thread 0:\n  store x 42 relaxed\n  store flag 1 release\n\n\
                     Thread 1:\n  load r0 flag acquire\n  load r1 x relaxed\n\n\
                     outcome: 1:r0=1, 1:r1=0 -> Forbidden",
                    "C++11-MP",
                ),
                ShotExample::new(
                    "C++11 acq_rel RMW",
                    "Generate a test using acq_rel RMW for mutual exclusion",
                    "test Cpp11_AcqRelRMW\ninit lock = 0\ninit data = 0\n\n\
                     Thread 0:\n  rmw r0 lock 1 acq_rel\n  store data 42 relaxed\n  store lock 0 release\n\n\
                     Thread 1:\n  rmw r0 lock 1 acq_rel\n  load r1 data relaxed\n  store lock 0 release\n\n\
                     outcome: 0:r0=0, 1:r0=0 -> Forbidden",
                    "C++11-RMW",
                ),
            ],
            key_properties: vec![
                "Relaxed: no ordering guarantees".to_string(),
                "Acquire/Release: creates synchronises-with edges".to_string(),
                "Release sequence: extends through RMW chain".to_string(),
                "SeqCst: total order on all SC operations".to_string(),
                "Mixed SC and non-SC can be subtle".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Bug pattern prompt
// ---------------------------------------------------------------------------

/// Prompts designed to generate tests that expose a specific bug class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BugPatternPrompt {
    pub bug_class: String,
    pub description: String,
    pub system_instruction: String,
    pub examples: Vec<ShotExample>,
    pub required_orderings: Vec<Ordering>,
    pub typical_thread_count: usize,
}

impl BugPatternPrompt {
    /// Build a flat prompt for this bug pattern.
    pub fn render_prompt(&self, additional_context: &str) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(&self.system_instruction);
        out.push_str("\n\n");

        out.push_str(&format!("Bug class: {}\n", self.bug_class));
        out.push_str(&format!("Typical thread count: {}\n", self.typical_thread_count));
        out.push_str(&format!(
            "Required orderings: {:?}\n\n",
            self.required_orderings
        ));

        for (i, ex) in self.examples.iter().enumerate() {
            out.push_str(&format!("--- Example {} ---\n", i + 1));
            out.push_str(&ex.render());
            out.push('\n');
        }

        if !additional_context.is_empty() {
            out.push_str("\nAdditional context:\n");
            out.push_str(additional_context);
            out.push('\n');
        }

        out
    }

    /// Number of curated examples.
    pub fn example_count(&self) -> usize {
        self.examples.len()
    }
}

// ---------------------------------------------------------------------------
// Model-specific prompt
// ---------------------------------------------------------------------------

/// Prompts tailored to a specific memory model family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpecificPrompt {
    pub model_family: String,
    pub description: String,
    pub system_instruction: String,
    pub examples: Vec<ShotExample>,
    pub key_properties: Vec<String>,
}

impl ModelSpecificPrompt {
    /// Build a flat prompt for this model.
    pub fn render_prompt(&self, additional_context: &str) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(&self.system_instruction);
        out.push_str("\n\n");

        out.push_str("Key properties of this model:\n");
        for prop in &self.key_properties {
            out.push_str(&format!("  - {}\n", prop));
        }
        out.push('\n');

        for (i, ex) in self.examples.iter().enumerate() {
            out.push_str(&format!("--- Example {} ---\n", i + 1));
            out.push_str(&ex.render());
            out.push('\n');
        }

        if !additional_context.is_empty() {
            out.push_str("\nAdditional context:\n");
            out.push_str(additional_context);
            out.push('\n');
        }

        out
    }

    /// Number of curated examples.
    pub fn example_count(&self) -> usize {
        self.examples.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Library construction -----------------------------------------------

    #[test]
    fn test_library_new_has_all_patterns() {
        let lib = PatternPromptLibrary::new();
        let names = lib.pattern_names();
        assert!(names.contains(&"SB".to_string()));
        assert!(names.contains(&"MP".to_string()));
        assert!(names.contains(&"LB".to_string()));
        assert!(names.contains(&"IRIW".to_string()));
        assert!(names.contains(&"2+2W".to_string()));
        assert!(names.contains(&"RWC".to_string()));
        assert!(names.contains(&"WRC".to_string()));
    }

    #[test]
    fn test_library_default() {
        let lib = PatternPromptLibrary::default();
        assert!(!lib.pattern_prompts.is_empty());
    }

    // -- Pattern lookup -----------------------------------------------------

    #[test]
    fn test_get_pattern_case_insensitive() {
        let lib = PatternPromptLibrary::new();
        assert!(lib.get_pattern("sb").is_some());
        assert!(lib.get_pattern("SB").is_some());
        assert!(lib.get_pattern("Sb").is_some());
    }

    #[test]
    fn test_get_pattern_unknown() {
        let lib = PatternPromptLibrary::new();
        assert!(lib.get_pattern("NONEXISTENT").is_none());
    }

    #[test]
    fn test_pattern_entry_has_examples() {
        let lib = PatternPromptLibrary::new();
        for name in lib.pattern_names() {
            let entry = lib.get_pattern(&name).unwrap();
            assert!(
                entry.examples.len() >= 2,
                "pattern {} has fewer than 2 examples",
                name
            );
        }
    }

    #[test]
    fn test_pattern_entry_has_constraints() {
        let lib = PatternPromptLibrary::new();
        for name in lib.pattern_names() {
            let entry = lib.get_pattern(&name).unwrap();
            assert!(
                !entry.constraints.is_empty(),
                "pattern {} has no constraints",
                name
            );
        }
    }

    #[test]
    fn test_pattern_entry_has_system_instruction() {
        let lib = PatternPromptLibrary::new();
        for name in lib.pattern_names() {
            let entry = lib.get_pattern(&name).unwrap();
            assert!(
                !entry.system_instruction.is_empty(),
                "pattern {} has empty system instruction",
                name
            );
        }
    }

    // -- Bug-class prompts --------------------------------------------------

    #[test]
    fn test_bug_classes() {
        let lib = PatternPromptLibrary::new();
        let classes = lib.bug_classes();
        assert!(classes.contains(&"data-race".to_string()));
        assert!(classes.contains(&"use-after-free".to_string()));
        assert!(classes.contains(&"aba-problem".to_string()));
        assert!(classes.contains(&"ordering-violation".to_string()));
        assert!(classes.contains(&"publication-failure".to_string()));
    }

    #[test]
    fn test_get_bug_prompts() {
        let lib = PatternPromptLibrary::new();
        let prompts = lib.get_bug_prompts("data-race");
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].bug_class, "data-race");
    }

    #[test]
    fn test_get_bug_prompts_partial_match() {
        let lib = PatternPromptLibrary::new();
        let prompts = lib.get_bug_prompts("race");
        assert!(prompts.len() >= 1);
    }

    #[test]
    fn test_get_bug_prompts_unknown() {
        let lib = PatternPromptLibrary::new();
        let prompts = lib.get_bug_prompts("nonexistent");
        assert!(prompts.is_empty());
    }

    #[test]
    fn test_bug_prompt_has_examples() {
        let lib = PatternPromptLibrary::new();
        for bp in &lib.bug_prompts {
            assert!(
                bp.example_count() >= 2,
                "bug class {} has fewer than 2 examples",
                bp.bug_class
            );
        }
    }

    #[test]
    fn test_bug_prompt_render() {
        let lib = PatternPromptLibrary::new();
        let prompt = &lib.bug_prompts[0];
        let rendered = prompt.render_prompt("extra context");
        assert!(rendered.contains(&prompt.bug_class));
        assert!(rendered.contains("extra context"));
        assert!(rendered.contains("Example 1"));
    }

    #[test]
    fn test_bug_prompt_render_empty_context() {
        let lib = PatternPromptLibrary::new();
        let prompt = &lib.bug_prompts[0];
        let rendered = prompt.render_prompt("");
        assert!(!rendered.contains("Additional context"));
    }

    // -- Model-specific prompts ---------------------------------------------

    #[test]
    fn test_model_families() {
        let lib = PatternPromptLibrary::new();
        let tso = lib.get_model_prompts("TSO");
        assert_eq!(tso.len(), 1);
        let arm = lib.get_model_prompts("ARM");
        assert_eq!(arm.len(), 1);
        let gpu = lib.get_model_prompts("GPU");
        assert_eq!(gpu.len(), 1);
        let cpp = lib.get_model_prompts("C++11");
        assert_eq!(cpp.len(), 1);
    }

    #[test]
    fn test_model_prompt_has_examples() {
        let lib = PatternPromptLibrary::new();
        for mp in &lib.model_prompts {
            assert!(
                mp.example_count() >= 3,
                "model {} has fewer than 3 examples",
                mp.model_family
            );
        }
    }

    #[test]
    fn test_model_prompt_has_properties() {
        let lib = PatternPromptLibrary::new();
        for mp in &lib.model_prompts {
            assert!(
                !mp.key_properties.is_empty(),
                "model {} has no key properties",
                mp.model_family
            );
        }
    }

    #[test]
    fn test_model_prompt_render() {
        let lib = PatternPromptLibrary::new();
        let tso = &lib.model_prompts[0];
        let rendered = tso.render_prompt("TSO context");
        assert!(rendered.contains("TSO"));
        assert!(rendered.contains("Key properties"));
        assert!(rendered.contains("TSO context"));
    }

    #[test]
    fn test_model_prompt_render_empty_context() {
        let lib = PatternPromptLibrary::new();
        let tso = &lib.model_prompts[0];
        let rendered = tso.render_prompt("");
        assert!(!rendered.contains("Additional context"));
    }

    // -- TSO-specific -------------------------------------------------------

    #[test]
    fn test_tso_prompt_store_buffer() {
        let lib = PatternPromptLibrary::new();
        let tso = lib.get_model_prompts("TSO");
        let prompt = tso[0];
        assert!(prompt.system_instruction.contains("store buffer"));
        assert!(prompt.examples.iter().any(|e| e.pattern_tag.contains("SB")));
    }

    #[test]
    fn test_tso_prompt_properties() {
        let lib = PatternPromptLibrary::new();
        let tso = lib.get_model_prompts("TSO");
        let prompt = tso[0];
        assert!(prompt
            .key_properties
            .iter()
            .any(|p| p.contains("store-load") || p.contains("Store-load")));
    }

    // -- ARM/RISC-V-specific ------------------------------------------------

    #[test]
    fn test_arm_prompt_multi_copy() {
        let lib = PatternPromptLibrary::new();
        let arm = lib.get_model_prompts("ARM");
        let prompt = arm[0];
        assert!(prompt.system_instruction.contains("multi-copy"));
    }

    #[test]
    fn test_arm_prompt_dependencies() {
        let lib = PatternPromptLibrary::new();
        let arm = lib.get_model_prompts("ARM");
        let prompt = arm[0];
        assert!(prompt
            .key_properties
            .iter()
            .any(|p| p.to_lowercase().contains("dependenc")));
    }

    // -- GPU-specific -------------------------------------------------------

    #[test]
    fn test_gpu_prompt_scopes() {
        let lib = PatternPromptLibrary::new();
        let gpu = lib.get_model_prompts("GPU");
        let prompt = gpu[0];
        assert!(prompt.system_instruction.contains("scope"));
        assert!(prompt
            .key_properties
            .iter()
            .any(|p| p.contains("CTA")));
    }

    #[test]
    fn test_gpu_examples_have_scoped_orderings() {
        let lib = PatternPromptLibrary::new();
        let gpu = lib.get_model_prompts("GPU");
        let prompt = gpu[0];
        let has_scoped = prompt.examples.iter().any(|e| {
            e.output.contains("release_cta")
                || e.output.contains("acquire_cta")
                || e.output.contains("release_system")
                || e.output.contains("acquire_system")
        });
        assert!(has_scoped, "GPU examples should use scoped orderings");
    }

    // -- C++11-specific -----------------------------------------------------

    #[test]
    fn test_cpp11_release_sequence() {
        let lib = PatternPromptLibrary::new();
        let cpp = lib.get_model_prompts("C++11");
        let prompt = cpp[0];
        assert!(prompt
            .examples
            .iter()
            .any(|e| e.pattern_tag.contains("relseq")));
        assert!(prompt.system_instruction.contains("Release sequence")
            || prompt.system_instruction.contains("release sequence"));
    }

    #[test]
    fn test_cpp11_has_sc_example() {
        let lib = PatternPromptLibrary::new();
        let cpp = lib.get_model_prompts("C++11");
        let prompt = cpp[0];
        assert!(prompt.examples.iter().any(|e| e.output.contains("seq_cst")));
    }

    // -- ShotExample render -------------------------------------------------

    #[test]
    fn test_shot_example_render_contains_fields() {
        let ex = ShotExample::new("desc", "input text", "output text", "TAG");
        let rendered = ex.render();
        assert!(rendered.contains("desc"));
        assert!(rendered.contains("input text"));
        assert!(rendered.contains("output text"));
        assert!(rendered.contains("TAG"));
    }

    // -- Pattern entry descriptions -----------------------------------------

    #[test]
    fn test_sb_description() {
        let lib = PatternPromptLibrary::new();
        let sb = lib.get_pattern("SB").unwrap();
        assert!(sb.description.contains("Store-Buffering"));
    }

    #[test]
    fn test_mp_description() {
        let lib = PatternPromptLibrary::new();
        let mp = lib.get_pattern("MP").unwrap();
        assert!(mp.description.contains("Message-Passing"));
    }

    #[test]
    fn test_iriw_description() {
        let lib = PatternPromptLibrary::new();
        let iriw = lib.get_pattern("IRIW").unwrap();
        assert!(iriw.description.contains("Independent"));
    }

    // -- All examples parse correctly with tag matching ---------------------

    #[test]
    fn test_all_examples_have_matching_tags() {
        let lib = PatternPromptLibrary::new();
        for (name, entry) in &lib.pattern_prompts {
            for (i, ex) in entry.examples.iter().enumerate() {
                assert!(
                    !ex.pattern_tag.is_empty(),
                    "pattern {} example {} has empty tag",
                    name,
                    i
                );
            }
        }
    }

    #[test]
    fn test_all_bug_examples_have_tags() {
        let lib = PatternPromptLibrary::new();
        for bp in &lib.bug_prompts {
            for (i, ex) in bp.examples.iter().enumerate() {
                assert!(
                    !ex.pattern_tag.is_empty(),
                    "bug class {} example {} has empty tag",
                    bp.bug_class,
                    i
                );
            }
        }
    }

    #[test]
    fn test_all_model_examples_have_tags() {
        let lib = PatternPromptLibrary::new();
        for mp in &lib.model_prompts {
            for (i, ex) in mp.examples.iter().enumerate() {
                assert!(
                    !ex.pattern_tag.is_empty(),
                    "model {} example {} has empty tag",
                    mp.model_family,
                    i
                );
            }
        }
    }

    // -- Roundtrip: count total examples ------------------------------------

    #[test]
    fn test_total_example_count() {
        let lib = PatternPromptLibrary::new();
        let pattern_count: usize = lib.pattern_prompts.values().map(|e| e.examples.len()).sum();
        let bug_count: usize = lib.bug_prompts.iter().map(|b| b.example_count()).sum();
        let model_count: usize = lib.model_prompts.iter().map(|m| m.example_count()).sum();
        let total = pattern_count + bug_count + model_count;
        // We have a rich library
        assert!(
            total >= 40,
            "expected at least 40 total examples, got {}",
            total
        );
    }
}
