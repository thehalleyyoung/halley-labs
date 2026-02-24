//! Standard litmus test catalog for LITMUS∞.
//!
//! Provides a comprehensive catalog of standard litmus test patterns:
//! MP (Message Passing), SB (Store Buffering), LB (Load Buffering),
//! IRIW, WRC, ISA2, R, S, 2+2W, and GPU-specific scoped variants.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::checker::{
    LitmusTest, Thread, Instruction, Outcome, LitmusOutcome,
    Address, Value,
};
use crate::checker::litmus::{Ordering, Scope, RegId};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X: Address = 0x100;
const Y: Address = 0x200;
const Z: Address = 0x300;
const W: Address = 0x400;

// ---------------------------------------------------------------------------
// PatternKind
// ---------------------------------------------------------------------------

/// The kind of litmus test pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternKind {
    /// Message Passing.
    MP,
    /// Store Buffering (Dekker).
    SB,
    /// Load Buffering.
    LB,
    /// Independent Reads of Independent Writes.
    IRIW,
    /// Write-Read Coherence.
    WRC,
    /// ISA2 (3-thread chain).
    ISA2,
    /// 2+2W (coherence writes).
    TwoPlusTwoW,
    /// R pattern (read-read coherence).
    R,
    /// S pattern (store atomicity).
    S,
    /// RMW-based patterns.
    RMW,
    /// CoRR (coherence of read-read).
    CoRR,
    /// CoWW (coherence of write-write).
    CoWW,
    /// CoRW (coherence of read-write).
    CoRW,
    /// CoWR (coherence of write-read).
    CoWR,
}

impl fmt::Display for PatternKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MP => write!(f, "MP"),
            Self::SB => write!(f, "SB"),
            Self::LB => write!(f, "LB"),
            Self::IRIW => write!(f, "IRIW"),
            Self::WRC => write!(f, "WRC"),
            Self::ISA2 => write!(f, "ISA2"),
            Self::TwoPlusTwoW => write!(f, "2+2W"),
            Self::R => write!(f, "R"),
            Self::S => write!(f, "S"),
            Self::RMW => write!(f, "RMW"),
            Self::CoRR => write!(f, "CoRR"),
            Self::CoWW => write!(f, "CoWW"),
            Self::CoRW => write!(f, "CoRW"),
            Self::CoWR => write!(f, "CoWR"),
        }
    }
}

impl PatternKind {
    /// All pattern kinds.
    pub fn all() -> Vec<Self> {
        vec![
            Self::MP, Self::SB, Self::LB, Self::IRIW,
            Self::WRC, Self::ISA2, Self::TwoPlusTwoW,
            Self::R, Self::S, Self::RMW,
            Self::CoRR, Self::CoWW, Self::CoRW, Self::CoWR,
        ]
    }

    /// Minimum number of threads for this pattern.
    pub fn min_threads(&self) -> usize {
        match self {
            Self::MP | Self::SB | Self::LB | Self::R |
            Self::CoRR | Self::CoWW | Self::CoRW | Self::CoWR |
            Self::TwoPlusTwoW | Self::RMW => 2,
            Self::WRC | Self::ISA2 | Self::S => 3,
            Self::IRIW => 4,
        }
    }

    /// Short description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MP => "Message Passing: W(x);W(y) || R(y);R(x)",
            Self::SB => "Store Buffering: W(x);R(y) || W(y);R(x)",
            Self::LB => "Load Buffering: R(x);W(y) || R(y);W(x)",
            Self::IRIW => "Independent Reads of Independent Writes",
            Self::WRC => "Write-Read Coherence chain",
            Self::ISA2 => "ISA2 three-thread chain",
            Self::TwoPlusTwoW => "2+2W coherence writes",
            Self::R => "Read-read coherence",
            Self::S => "Store atomicity",
            Self::RMW => "Read-modify-write patterns",
            Self::CoRR => "Coherence of read-read",
            Self::CoWW => "Coherence of write-write",
            Self::CoRW => "Coherence of read-write",
            Self::CoWR => "Coherence of write-read",
        }
    }
}

// ---------------------------------------------------------------------------
// CatalogEntry
// ---------------------------------------------------------------------------

/// An entry in the test catalog.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Test name.
    pub name: String,
    /// Pattern kind.
    pub pattern: PatternKind,
    /// The litmus test.
    pub test: LitmusTest,
    /// Description.
    pub description: String,
    /// Tags for filtering.
    pub tags: Vec<String>,
    /// Which models forbid the interesting outcome.
    pub forbidden_under: Vec<String>,
    /// Which models allow the interesting outcome.
    pub allowed_under: Vec<String>,
}

impl CatalogEntry {
    /// Create a new catalog entry.
    pub fn new(name: &str, pattern: PatternKind, test: LitmusTest) -> Self {
        Self {
            name: name.into(),
            pattern,
            test,
            description: String::new(),
            tags: Vec::new(),
            forbidden_under: Vec::new(),
            allowed_under: Vec::new(),
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Mark as forbidden under a model.
    pub fn forbidden_under_model(mut self, model: &str) -> Self {
        self.forbidden_under.push(model.into());
        self
    }

    /// Mark as allowed under a model.
    pub fn allowed_under_model(mut self, model: &str) -> Self {
        self.allowed_under.push(model.into());
        self
    }
}

impl fmt::Display for CatalogEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({})", self.pattern, self.name, self.description)
    }
}

// ---------------------------------------------------------------------------
// TestCatalog
// ---------------------------------------------------------------------------

/// A catalog of litmus tests organised by pattern.
#[derive(Debug, Clone)]
pub struct TestCatalog {
    entries: Vec<CatalogEntry>,
}

impl TestCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Create a catalog with all standard tests.
    pub fn standard() -> Self {
        let mut cat = Self::new();
        cat.add_mp_variants();
        cat.add_sb_variants();
        cat.add_lb_variants();
        cat.add_iriw_variants();
        cat.add_wrc_variants();
        cat.add_isa2_variants();
        cat.add_two_plus_two_w_variants();
        cat.add_r_variants();
        cat.add_s_variants();
        cat.add_coherence_variants();
        cat
    }

    /// Create a catalog with GPU tests included.
    pub fn with_gpu() -> Self {
        let mut cat = Self::standard();
        cat.add_gpu_mp_variants();
        cat.add_gpu_sb_variants();
        cat.add_gpu_lb_variants();
        cat
    }

    /// Add an entry.
    pub fn add(&mut self, entry: CatalogEntry) {
        self.entries.push(entry);
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries.
    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    /// Get all tests.
    pub fn tests(&self) -> Vec<&LitmusTest> {
        self.entries.iter().map(|e| &e.test).collect()
    }

    /// Get entries by pattern kind.
    pub fn by_pattern(&self, pattern: PatternKind) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.pattern == pattern).collect()
    }

    /// Get entries by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| e.tags.contains(&tag.to_string())).collect()
    }

    /// Get entries forbidden under a specific model.
    pub fn forbidden_under(&self, model: &str) -> Vec<&CatalogEntry> {
        self.entries.iter()
            .filter(|e| e.forbidden_under.contains(&model.to_string()))
            .collect()
    }

    /// Get entries allowed under a specific model.
    pub fn allowed_under(&self, model: &str) -> Vec<&CatalogEntry> {
        self.entries.iter()
            .filter(|e| e.allowed_under.contains(&model.to_string()))
            .collect()
    }

    /// Filter entries by a predicate on the test.
    pub fn filter<F: Fn(&LitmusTest) -> bool>(&self, pred: F) -> Vec<&CatalogEntry> {
        self.entries.iter().filter(|e| pred(&e.test)).collect()
    }

    /// Summary statistics.
    pub fn summary(&self) -> CatalogSummary {
        let mut by_pattern: HashMap<PatternKind, usize> = HashMap::new();
        let mut by_tag: HashMap<String, usize> = HashMap::new();

        for entry in &self.entries {
            *by_pattern.entry(entry.pattern).or_default() += 1;
            for tag in &entry.tags {
                *by_tag.entry(tag.clone()).or_default() += 1;
            }
        }

        CatalogSummary {
            total: self.entries.len(),
            by_pattern,
            by_tag,
        }
    }

    // -----------------------------------------------------------------------
    // MP variants
    // -----------------------------------------------------------------------

    fn add_mp_variants(&mut self) {
        // MP-relaxed (base)
        self.add(CatalogEntry::new("MP", PatternKind::MP, mp_relaxed())
            .with_description("Message Passing, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC")
            .allowed_under_model("TSO")
            .allowed_under_model("PSO"));

        // MP with release-acquire
        self.add(CatalogEntry::new("MP-ra", PatternKind::MP, mp_release_acquire())
            .with_description("Message Passing with release-acquire")
            .with_tag("release-acquire")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO"));

        // MP with seq-cst
        self.add(CatalogEntry::new("MP-sc", PatternKind::MP, mp_seq_cst())
            .with_description("Message Passing with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));

        // MP with fence
        self.add(CatalogEntry::new("MP-fence", PatternKind::MP, mp_fenced())
            .with_description("Message Passing with fences")
            .with_tag("fenced")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO"));

        // MP with acquire on reads only
        self.add(CatalogEntry::new("MP-acq-read", PatternKind::MP, mp_acquire_reads())
            .with_description("MP with acquire on reader thread")
            .with_tag("partial-ordering"));

        // MP with release on writes only
        self.add(CatalogEntry::new("MP-rel-write", PatternKind::MP, mp_release_writes())
            .with_description("MP with release on writer thread")
            .with_tag("partial-ordering"));

        // MP with 3 locations
        self.add(CatalogEntry::new("MP-3loc", PatternKind::MP, mp_three_locations())
            .with_description("MP with 3 memory locations")
            .with_tag("multi-location"));
    }

    // -----------------------------------------------------------------------
    // SB variants
    // -----------------------------------------------------------------------

    fn add_sb_variants(&mut self) {
        // SB-relaxed
        self.add(CatalogEntry::new("SB", PatternKind::SB, sb_relaxed())
            .with_description("Store Buffering, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC")
            .allowed_under_model("TSO")
            .allowed_under_model("PSO"));

        // SB with seq-cst
        self.add(CatalogEntry::new("SB-sc", PatternKind::SB, sb_seq_cst())
            .with_description("Store Buffering with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));

        // SB with fences
        self.add(CatalogEntry::new("SB-fence", PatternKind::SB, sb_fenced())
            .with_description("Store Buffering with fences between W and R")
            .with_tag("fenced")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO"));

        // SB with RMW
        self.add(CatalogEntry::new("SB-rmw", PatternKind::SB, sb_with_rmw())
            .with_description("Store Buffering with RMW replacing load")
            .with_tag("rmw"));

        // SB asymmetric
        self.add(CatalogEntry::new("SB-asym", PatternKind::SB, sb_asymmetric())
            .with_description("SB with different orderings per thread")
            .with_tag("asymmetric"));
    }

    // -----------------------------------------------------------------------
    // LB variants
    // -----------------------------------------------------------------------

    fn add_lb_variants(&mut self) {
        // LB-relaxed
        self.add(CatalogEntry::new("LB", PatternKind::LB, lb_relaxed())
            .with_description("Load Buffering, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO"));

        // LB with acquire
        self.add(CatalogEntry::new("LB-acq", PatternKind::LB, lb_acquire())
            .with_description("Load Buffering with acquire loads")
            .with_tag("acquire")
            .forbidden_under_model("SC"));

        // LB with release
        self.add(CatalogEntry::new("LB-rel", PatternKind::LB, lb_release())
            .with_description("Load Buffering with release stores")
            .with_tag("release")
            .forbidden_under_model("SC"));

        // LB with seq-cst
        self.add(CatalogEntry::new("LB-sc", PatternKind::LB, lb_seq_cst())
            .with_description("Load Buffering with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));

        // LB with fences
        self.add(CatalogEntry::new("LB-fence", PatternKind::LB, lb_fenced())
            .with_description("Load Buffering with fences")
            .with_tag("fenced")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // IRIW variants
    // -----------------------------------------------------------------------

    fn add_iriw_variants(&mut self) {
        // IRIW-relaxed
        self.add(CatalogEntry::new("IRIW", PatternKind::IRIW, iriw_relaxed())
            .with_description("Independent Reads of Independent Writes, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC")
            .allowed_under_model("TSO"));

        // IRIW with seq-cst
        self.add(CatalogEntry::new("IRIW-sc", PatternKind::IRIW, iriw_seq_cst())
            .with_description("IRIW with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));

        // IRIW with acquire reads
        self.add(CatalogEntry::new("IRIW-acq", PatternKind::IRIW, iriw_acquire())
            .with_description("IRIW with acquire reads")
            .with_tag("acquire"));
    }

    // -----------------------------------------------------------------------
    // WRC variants
    // -----------------------------------------------------------------------

    fn add_wrc_variants(&mut self) {
        // WRC-relaxed
        self.add(CatalogEntry::new("WRC", PatternKind::WRC, wrc_relaxed())
            .with_description("Write-Read Coherence, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC"));

        // WRC with release-acquire
        self.add(CatalogEntry::new("WRC-ra", PatternKind::WRC, wrc_release_acquire())
            .with_description("WRC with release-acquire ordering")
            .with_tag("release-acquire")
            .forbidden_under_model("SC"));

        // WRC with seq-cst
        self.add(CatalogEntry::new("WRC-sc", PatternKind::WRC, wrc_seq_cst())
            .with_description("WRC with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // ISA2 variants
    // -----------------------------------------------------------------------

    fn add_isa2_variants(&mut self) {
        // ISA2-relaxed
        self.add(CatalogEntry::new("ISA2", PatternKind::ISA2, isa2_relaxed())
            .with_description("ISA2 three-thread chain, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC"));

        // ISA2 with release-acquire
        self.add(CatalogEntry::new("ISA2-ra", PatternKind::ISA2, isa2_release_acquire())
            .with_description("ISA2 with release-acquire")
            .with_tag("release-acquire")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // 2+2W variants
    // -----------------------------------------------------------------------

    fn add_two_plus_two_w_variants(&mut self) {
        // 2+2W-relaxed
        self.add(CatalogEntry::new("2+2W", PatternKind::TwoPlusTwoW, two_plus_two_w_relaxed())
            .with_description("2+2W coherence writes, all relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC"));

        // 2+2W with seq-cst
        self.add(CatalogEntry::new("2+2W-sc", PatternKind::TwoPlusTwoW, two_plus_two_w_seq_cst())
            .with_description("2+2W with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // R variants
    // -----------------------------------------------------------------------

    fn add_r_variants(&mut self) {
        // R-relaxed
        self.add(CatalogEntry::new("R", PatternKind::R, r_relaxed())
            .with_description("Read-read coherence, relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO")
            .forbidden_under_model("PSO"));

        // R with acquire
        self.add(CatalogEntry::new("R-acq", PatternKind::R, r_acquire())
            .with_description("Read-read coherence with acquire")
            .with_tag("acquire")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // S variants
    // -----------------------------------------------------------------------

    fn add_s_variants(&mut self) {
        // S-relaxed
        self.add(CatalogEntry::new("S", PatternKind::S, s_relaxed())
            .with_description("Store atomicity, relaxed")
            .with_tag("basic")
            .forbidden_under_model("SC"));

        // S with seq-cst
        self.add(CatalogEntry::new("S-sc", PatternKind::S, s_seq_cst())
            .with_description("Store atomicity with SeqCst")
            .with_tag("seq-cst")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // Coherence variants
    // -----------------------------------------------------------------------

    fn add_coherence_variants(&mut self) {
        // CoRR
        self.add(CatalogEntry::new("CoRR", PatternKind::CoRR, corr_test())
            .with_description("Coherence of read-read on same location")
            .with_tag("coherence")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO")
            .forbidden_under_model("PSO"));

        // CoWW
        self.add(CatalogEntry::new("CoWW", PatternKind::CoWW, coww_test())
            .with_description("Coherence of write-write on same location")
            .with_tag("coherence")
            .forbidden_under_model("SC")
            .forbidden_under_model("TSO")
            .forbidden_under_model("PSO"));

        // CoRW
        self.add(CatalogEntry::new("CoRW", PatternKind::CoRW, corw_test())
            .with_description("Coherence of read-write")
            .with_tag("coherence")
            .forbidden_under_model("SC"));

        // CoWR
        self.add(CatalogEntry::new("CoWR", PatternKind::CoWR, cowr_test())
            .with_description("Coherence of write-read")
            .with_tag("coherence")
            .forbidden_under_model("SC"));
    }

    // -----------------------------------------------------------------------
    // GPU variants
    // -----------------------------------------------------------------------

    fn add_gpu_mp_variants(&mut self) {
        for scope in &[Scope::CTA, Scope::GPU, Scope::System] {
            let name = format!("MP-gpu-{}", scope);
            let test = mp_gpu_scoped(*scope);
            self.add(CatalogEntry::new(&name, PatternKind::MP, test)
                .with_description(&format!("MP with {} scope", scope))
                .with_tag("gpu")
                .with_tag(&format!("scope-{}", scope)));
        }

        // Cross-scope MP
        self.add(CatalogEntry::new("MP-cross-cta-gpu", PatternKind::MP,
            mp_cross_scope(Scope::CTA, Scope::GPU))
            .with_description("MP with CTA writer, GPU reader")
            .with_tag("gpu")
            .with_tag("cross-scope"));

        self.add(CatalogEntry::new("MP-cross-gpu-sys", PatternKind::MP,
            mp_cross_scope(Scope::GPU, Scope::System))
            .with_description("MP with GPU writer, System reader")
            .with_tag("gpu")
            .with_tag("cross-scope"));
    }

    fn add_gpu_sb_variants(&mut self) {
        for scope in &[Scope::CTA, Scope::GPU, Scope::System] {
            let name = format!("SB-gpu-{}", scope);
            let test = sb_gpu_scoped(*scope);
            self.add(CatalogEntry::new(&name, PatternKind::SB, test)
                .with_description(&format!("SB with {} scope", scope))
                .with_tag("gpu")
                .with_tag(&format!("scope-{}", scope)));
        }
    }

    fn add_gpu_lb_variants(&mut self) {
        for scope in &[Scope::CTA, Scope::GPU, Scope::System] {
            let name = format!("LB-gpu-{}", scope);
            let test = lb_gpu_scoped(*scope);
            self.add(CatalogEntry::new(&name, PatternKind::LB, test)
                .with_description(&format!("LB with {} scope", scope))
                .with_tag("gpu")
                .with_tag(&format!("scope-{}", scope)));
        }
    }
}

impl Default for TestCatalog {
    fn default() -> Self {
        Self::standard()
    }
}

impl fmt::Display for TestCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Test Catalog ({} entries)", self.entries.len())?;
        let summary = self.summary();
        for (pattern, count) in &summary.by_pattern {
            writeln!(f, "  {}: {} tests", pattern, count)?;
        }
        Ok(())
    }
}

/// Summary statistics for a catalog.
#[derive(Debug, Clone)]
pub struct CatalogSummary {
    pub total: usize,
    pub by_pattern: HashMap<PatternKind, usize>,
    pub by_tag: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// MP test factories
// ---------------------------------------------------------------------------

/// MP with all relaxed.
fn mp_relaxed() -> LitmusTest {
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

/// MP with release-acquire.
fn mp_release_acquire() -> LitmusTest {
    let mut test = LitmusTest::new("MP-ra");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Release);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Acquire);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// MP with SeqCst.
fn mp_seq_cst() -> LitmusTest {
    let mut test = LitmusTest::new("MP-sc");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::SeqCst);
    t0.store(Y, 1, Ordering::SeqCst);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::SeqCst);
    t1.load(1, X, Ordering::SeqCst);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// MP with fences.
fn mp_fenced() -> LitmusTest {
    let mut test = LitmusTest::new("MP-fence");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.fence(Ordering::SeqCst, Scope::None);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.fence(Ordering::SeqCst, Scope::None);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// MP with acquire on reader.
fn mp_acquire_reads() -> LitmusTest {
    let mut test = LitmusTest::new("MP-acq-read");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Acquire);
    t1.load(1, X, Ordering::Acquire);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// MP with release on writer.
fn mp_release_writes() -> LitmusTest {
    let mut test = LitmusTest::new("MP-rel-write");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Release);
    t0.store(Y, 1, Ordering::Release);
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

/// MP with three locations.
fn mp_three_locations() -> LitmusTest {
    let mut test = LitmusTest::new("MP-3loc");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);
    test.set_initial(Z, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Relaxed);
    t0.store(Z, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Z, Ordering::Relaxed);
    t1.load(1, Y, Ordering::Relaxed);
    t1.load(2, X, Ordering::Relaxed);
    test.add_thread(t1);

    test
}

// ---------------------------------------------------------------------------
// SB test factories
// ---------------------------------------------------------------------------

fn sb_relaxed() -> LitmusTest {
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

fn sb_seq_cst() -> LitmusTest {
    let mut test = LitmusTest::new("SB-sc");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::SeqCst);
    t0.load(0, Y, Ordering::SeqCst);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::SeqCst);
    t1.load(0, X, Ordering::SeqCst);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn sb_fenced() -> LitmusTest {
    let mut test = LitmusTest::new("SB-fence");
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

fn sb_with_rmw() -> LitmusTest {
    let mut test = LitmusTest::new("SB-rmw");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.rmw(0, Y, 0, Ordering::SeqCst);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    t1.rmw(0, X, 0, Ordering::SeqCst);
    test.add_thread(t1);

    test
}

fn sb_asymmetric() -> LitmusTest {
    let mut test = LitmusTest::new("SB-asym");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::SeqCst);
    t0.load(0, Y, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, Ordering::Relaxed);
    t1.load(0, X, Ordering::SeqCst);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

// ---------------------------------------------------------------------------
// LB test factories
// ---------------------------------------------------------------------------

fn lb_relaxed() -> LitmusTest {
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

fn lb_acquire() -> LitmusTest {
    let mut test = LitmusTest::new("LB-acq");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, Ordering::Acquire);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Acquire);
    t1.store(X, 1, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn lb_release() -> LitmusTest {
    let mut test = LitmusTest::new("LB-rel");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Release);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.store(X, 1, Ordering::Release);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn lb_seq_cst() -> LitmusTest {
    let mut test = LitmusTest::new("LB-sc");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, Ordering::SeqCst);
    t0.store(Y, 1, Ordering::SeqCst);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::SeqCst);
    t1.store(X, 1, Ordering::SeqCst);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn lb_fenced() -> LitmusTest {
    let mut test = LitmusTest::new("LB-fence");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, Ordering::Relaxed);
    t0.fence(Ordering::SeqCst, Scope::None);
    t0.store(Y, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Relaxed);
    t1.fence(Ordering::SeqCst, Scope::None);
    t1.store(X, 1, Ordering::Relaxed);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

// ---------------------------------------------------------------------------
// IRIW test factories
// ---------------------------------------------------------------------------

fn iriw_relaxed() -> LitmusTest {
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

    // Forbidden under SC: T2 sees x=1,y=0 but T3 sees y=1,x=0
    test.expect(
        Outcome::new()
            .with_reg(2, 0, 1).with_reg(2, 1, 0)
            .with_reg(3, 0, 1).with_reg(3, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn iriw_seq_cst() -> LitmusTest {
    let mut test = iriw_relaxed();
    test.name = "IRIW-sc".into();
    // Replace all orderings with SeqCst.
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            match instr {
                Instruction::Load { ordering, .. } => *ordering = Ordering::SeqCst,
                Instruction::Store { ordering, .. } => *ordering = Ordering::SeqCst,
                _ => {}
            }
        }
    }
    test
}

fn iriw_acquire() -> LitmusTest {
    let mut test = iriw_relaxed();
    test.name = "IRIW-acq".into();
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            if let Instruction::Load { ordering, .. } = instr {
                *ordering = Ordering::Acquire;
            }
        }
    }
    test
}

// ---------------------------------------------------------------------------
// WRC test factories
// ---------------------------------------------------------------------------

fn wrc_relaxed() -> LitmusTest {
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

fn wrc_release_acquire() -> LitmusTest {
    let mut test = LitmusTest::new("WRC-ra");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Release);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, X, Ordering::Acquire);
    t1.store(Y, 1, Ordering::Release);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, Y, Ordering::Acquire);
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

fn wrc_seq_cst() -> LitmusTest {
    let mut test = wrc_relaxed();
    test.name = "WRC-sc".into();
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            match instr {
                Instruction::Load { ordering, .. } => *ordering = Ordering::SeqCst,
                Instruction::Store { ordering, .. } => *ordering = Ordering::SeqCst,
                _ => {}
            }
        }
    }
    test
}

// ---------------------------------------------------------------------------
// ISA2 test factories
// ---------------------------------------------------------------------------

fn isa2_relaxed() -> LitmusTest {
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

fn isa2_release_acquire() -> LitmusTest {
    let mut test = LitmusTest::new("ISA2-ra");
    test.set_initial(X, 0);
    test.set_initial(Y, 0);
    test.set_initial(Z, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(Y, 1, Ordering::Release);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, Ordering::Acquire);
    t1.store(Z, 1, Ordering::Release);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, Z, Ordering::Acquire);
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

// ---------------------------------------------------------------------------
// 2+2W test factories
// ---------------------------------------------------------------------------

fn two_plus_two_w_relaxed() -> LitmusTest {
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

    test
}

fn two_plus_two_w_seq_cst() -> LitmusTest {
    let mut test = two_plus_two_w_relaxed();
    test.name = "2+2W-sc".into();
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            if let Instruction::Store { ordering, .. } = instr {
                *ordering = Ordering::SeqCst;
            }
        }
    }
    test
}

// ---------------------------------------------------------------------------
// R pattern test factories
// ---------------------------------------------------------------------------

fn r_relaxed() -> LitmusTest {
    let mut test = LitmusTest::new("R");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(X, 2, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, X, Ordering::Relaxed);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    // Forbidden: read x=2 then x=1 (violates coherence)
    test.expect(
        Outcome::new().with_reg(1, 0, 2).with_reg(1, 1, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

fn r_acquire() -> LitmusTest {
    let mut test = r_relaxed();
    test.name = "R-acq".into();
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            if let Instruction::Load { ordering, .. } = instr {
                *ordering = Ordering::Acquire;
            }
        }
    }
    test
}

// ---------------------------------------------------------------------------
// S pattern test factories
// ---------------------------------------------------------------------------

fn s_relaxed() -> LitmusTest {
    let mut test = LitmusTest::new("S");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(X, 2, Ordering::Relaxed);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, X, Ordering::Relaxed);
    t2.load(1, X, Ordering::Relaxed);
    test.add_thread(t2);

    test
}

fn s_seq_cst() -> LitmusTest {
    let mut test = s_relaxed();
    test.name = "S-sc".into();
    for t in &mut test.threads {
        for instr in &mut t.instructions {
            match instr {
                Instruction::Load { ordering, .. } => *ordering = Ordering::SeqCst,
                Instruction::Store { ordering, .. } => *ordering = Ordering::SeqCst,
                _ => {}
            }
        }
    }
    test
}

// ---------------------------------------------------------------------------
// Coherence test factories
// ---------------------------------------------------------------------------

/// CoRR: Two reads of the same location should see consistent order.
fn corr_test() -> LitmusTest {
    let mut test = LitmusTest::new("CoRR");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(X, 2, Ordering::Relaxed);
    test.add_thread(t1);

    let mut t2 = Thread::new(2);
    t2.load(0, X, Ordering::Relaxed);
    t2.load(1, X, Ordering::Relaxed);
    test.add_thread(t2);

    // Forbidden: see x=2 then x=1
    test.expect(
        Outcome::new().with_reg(2, 0, 2).with_reg(2, 1, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

/// CoWW: Two writes to the same location are totally ordered.
fn coww_test() -> LitmusTest {
    let mut test = LitmusTest::new("CoWW");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.store(X, 2, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, X, Ordering::Relaxed);
    t1.load(1, X, Ordering::Relaxed);
    test.add_thread(t1);

    // Forbidden: read x=2 then x=1
    test.expect(
        Outcome::new().with_reg(1, 0, 2).with_reg(1, 1, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

/// CoRW: A read before a write to the same location.
fn corw_test() -> LitmusTest {
    let mut test = LitmusTest::new("CoRW");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, X, Ordering::Relaxed);
    t1.store(X, 2, Ordering::Relaxed);
    test.add_thread(t1);

    test
}

/// CoWR: A write before a read of the same location.
fn cowr_test() -> LitmusTest {
    let mut test = LitmusTest::new("CoWR");
    test.set_initial(X, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, Ordering::Relaxed);
    t0.load(0, X, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(X, 2, Ordering::Relaxed);
    test.add_thread(t1);

    test
}

// ---------------------------------------------------------------------------
// GPU-scoped test factories
// ---------------------------------------------------------------------------

fn scope_to_release(scope: Scope) -> Ordering {
    match scope {
        Scope::CTA => Ordering::ReleaseCTA,
        Scope::GPU => Ordering::ReleaseGPU,
        Scope::System => Ordering::ReleaseSystem,
        Scope::None => Ordering::Release,
    }
}

fn scope_to_acquire(scope: Scope) -> Ordering {
    match scope {
        Scope::CTA => Ordering::AcquireCTA,
        Scope::GPU => Ordering::AcquireGPU,
        Scope::System => Ordering::AcquireSystem,
        Scope::None => Ordering::Acquire,
    }
}

/// MP with GPU scoping.
fn mp_gpu_scoped(scope: Scope) -> LitmusTest {
    let w_ord = scope_to_release(scope);
    let r_ord = scope_to_acquire(scope);

    let mut test = LitmusTest::new(&format!("MP-gpu-{}", scope));
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, w_ord);
    t0.store(Y, 1, w_ord);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, r_ord);
    t1.load(1, X, r_ord);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// MP with cross-scope annotations.
fn mp_cross_scope(w_scope: Scope, r_scope: Scope) -> LitmusTest {
    let w_ord = scope_to_release(w_scope);
    let r_ord = scope_to_acquire(r_scope);

    let mut test = LitmusTest::new(&format!("MP-cross-{}-{}", w_scope, r_scope));
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, w_ord);
    t0.store(Y, 1, w_ord);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, r_ord);
    t1.load(1, X, r_ord);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// SB with GPU scoping.
fn sb_gpu_scoped(scope: Scope) -> LitmusTest {
    let w_ord = scope_to_release(scope);
    let r_ord = scope_to_acquire(scope);

    let mut test = LitmusTest::new(&format!("SB-gpu-{}", scope));
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.store(X, 1, w_ord);
    t0.load(0, Y, r_ord);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(Y, 1, w_ord);
    t1.load(0, X, r_ord);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

/// LB with GPU scoping.
fn lb_gpu_scoped(scope: Scope) -> LitmusTest {
    let r_ord = scope_to_acquire(scope);
    let w_ord = scope_to_release(scope);

    let mut test = LitmusTest::new(&format!("LB-gpu-{}", scope));
    test.set_initial(X, 0);
    test.set_initial(Y, 0);

    let mut t0 = Thread::new(0);
    t0.load(0, X, r_ord);
    t0.store(Y, 1, w_ord);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, Y, r_ord);
    t1.store(X, 1, w_ord);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 1).with_reg(1, 0, 1),
        LitmusOutcome::Forbidden,
    );
    test
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_kind_all() {
        let all = PatternKind::all();
        assert!(all.len() >= 10);
    }

    #[test]
    fn test_pattern_kind_display() {
        assert_eq!(format!("{}", PatternKind::MP), "MP");
        assert_eq!(format!("{}", PatternKind::SB), "SB");
        assert_eq!(format!("{}", PatternKind::IRIW), "IRIW");
    }

    #[test]
    fn test_pattern_kind_min_threads() {
        assert_eq!(PatternKind::MP.min_threads(), 2);
        assert_eq!(PatternKind::IRIW.min_threads(), 4);
        assert_eq!(PatternKind::WRC.min_threads(), 3);
    }

    #[test]
    fn test_pattern_kind_description() {
        let desc = PatternKind::MP.description();
        assert!(desc.contains("Message Passing"));
    }

    #[test]
    fn test_catalog_entry_creation() {
        let test = mp_relaxed();
        let entry = CatalogEntry::new("MP", PatternKind::MP, test)
            .with_description("test")
            .with_tag("basic")
            .forbidden_under_model("SC");
        assert_eq!(entry.name, "MP");
        assert_eq!(entry.pattern, PatternKind::MP);
        assert!(entry.tags.contains(&"basic".to_string()));
        assert!(entry.forbidden_under.contains(&"SC".to_string()));
    }

    #[test]
    fn test_catalog_entry_display() {
        let test = mp_relaxed();
        let entry = CatalogEntry::new("MP", PatternKind::MP, test)
            .with_description("Message Passing");
        let s = format!("{}", entry);
        assert!(s.contains("MP"));
    }

    #[test]
    fn test_empty_catalog() {
        let cat = TestCatalog::new();
        assert!(cat.is_empty());
        assert_eq!(cat.len(), 0);
    }

    #[test]
    fn test_standard_catalog() {
        let cat = TestCatalog::standard();
        assert!(!cat.is_empty());
        assert!(cat.len() > 20);
    }

    #[test]
    fn test_gpu_catalog() {
        let cat = TestCatalog::with_gpu();
        assert!(cat.len() > TestCatalog::standard().len());
    }

    #[test]
    fn test_catalog_by_pattern_mp() {
        let cat = TestCatalog::standard();
        let mp_tests = cat.by_pattern(PatternKind::MP);
        assert!(!mp_tests.is_empty());
        for entry in &mp_tests {
            assert_eq!(entry.pattern, PatternKind::MP);
        }
    }

    #[test]
    fn test_catalog_by_pattern_sb() {
        let cat = TestCatalog::standard();
        let sb_tests = cat.by_pattern(PatternKind::SB);
        assert!(!sb_tests.is_empty());
    }

    #[test]
    fn test_catalog_by_pattern_lb() {
        let cat = TestCatalog::standard();
        let lb_tests = cat.by_pattern(PatternKind::LB);
        assert!(!lb_tests.is_empty());
    }

    #[test]
    fn test_catalog_by_pattern_iriw() {
        let cat = TestCatalog::standard();
        let iriw_tests = cat.by_pattern(PatternKind::IRIW);
        assert!(!iriw_tests.is_empty());
    }

    #[test]
    fn test_catalog_by_pattern_wrc() {
        let cat = TestCatalog::standard();
        let wrc_tests = cat.by_pattern(PatternKind::WRC);
        assert!(!wrc_tests.is_empty());
    }

    #[test]
    fn test_catalog_by_tag() {
        let cat = TestCatalog::standard();
        let basic = cat.by_tag("basic");
        assert!(!basic.is_empty());
    }

    #[test]
    fn test_catalog_forbidden_under() {
        let cat = TestCatalog::standard();
        let sc_forbidden = cat.forbidden_under("SC");
        assert!(!sc_forbidden.is_empty());
    }

    #[test]
    fn test_catalog_tests() {
        let cat = TestCatalog::standard();
        let tests = cat.tests();
        assert_eq!(tests.len(), cat.len());
    }

    #[test]
    fn test_catalog_filter() {
        let cat = TestCatalog::standard();
        let two_thread = cat.filter(|t| t.thread_count() == 2);
        assert!(!two_thread.is_empty());
    }

    #[test]
    fn test_catalog_summary() {
        let cat = TestCatalog::standard();
        let summary = cat.summary();
        assert!(summary.total > 0);
        assert!(!summary.by_pattern.is_empty());
    }

    #[test]
    fn test_catalog_display() {
        let cat = TestCatalog::standard();
        let s = format!("{}", cat);
        assert!(s.contains("Test Catalog"));
    }

    #[test]
    fn test_mp_relaxed() {
        let test = mp_relaxed();
        assert_eq!(test.name, "MP");
        assert_eq!(test.thread_count(), 2);
        assert!(!test.expected_outcomes.is_empty());
    }

    #[test]
    fn test_mp_release_acquire() {
        let test = mp_release_acquire();
        assert_eq!(test.name, "MP-ra");
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_mp_seq_cst() {
        let test = mp_seq_cst();
        assert_eq!(test.name, "MP-sc");
    }

    #[test]
    fn test_mp_fenced() {
        let test = mp_fenced();
        assert!(test.total_instructions() > mp_relaxed().total_instructions());
    }

    #[test]
    fn test_mp_three_locations() {
        let test = mp_three_locations();
        assert!(test.all_addresses().len() >= 3);
    }

    #[test]
    fn test_sb_relaxed() {
        let test = sb_relaxed();
        assert_eq!(test.name, "SB");
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_sb_fenced() {
        let test = sb_fenced();
        assert!(test.total_instructions() > sb_relaxed().total_instructions());
    }

    #[test]
    fn test_sb_with_rmw() {
        let test = sb_with_rmw();
        let has_rmw = test.threads.iter().any(|t|
            t.instructions.iter().any(|i| matches!(i, Instruction::RMW { .. }))
        );
        assert!(has_rmw);
    }

    #[test]
    fn test_lb_relaxed() {
        let test = lb_relaxed();
        assert_eq!(test.name, "LB");
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_lb_fenced() {
        let test = lb_fenced();
        assert!(test.total_instructions() > lb_relaxed().total_instructions());
    }

    #[test]
    fn test_iriw_relaxed() {
        let test = iriw_relaxed();
        assert_eq!(test.name, "IRIW");
        assert_eq!(test.thread_count(), 4);
    }

    #[test]
    fn test_iriw_seq_cst() {
        let test = iriw_seq_cst();
        assert_eq!(test.name, "IRIW-sc");
    }

    #[test]
    fn test_wrc_relaxed() {
        let test = wrc_relaxed();
        assert_eq!(test.thread_count(), 3);
    }

    #[test]
    fn test_wrc_release_acquire() {
        let test = wrc_release_acquire();
        assert_eq!(test.name, "WRC-ra");
    }

    #[test]
    fn test_isa2_relaxed() {
        let test = isa2_relaxed();
        assert_eq!(test.thread_count(), 3);
        assert!(test.all_addresses().len() >= 3);
    }

    #[test]
    fn test_two_plus_two_w_relaxed() {
        let test = two_plus_two_w_relaxed();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_r_relaxed() {
        let test = r_relaxed();
        assert_eq!(test.thread_count(), 2);
        assert!(!test.expected_outcomes.is_empty());
    }

    #[test]
    fn test_s_relaxed() {
        let test = s_relaxed();
        assert_eq!(test.thread_count(), 3);
    }

    #[test]
    fn test_corr() {
        let test = corr_test();
        assert_eq!(test.thread_count(), 3);
        assert!(!test.expected_outcomes.is_empty());
    }

    #[test]
    fn test_coww() {
        let test = coww_test();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_corw() {
        let test = corw_test();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_cowr() {
        let test = cowr_test();
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_mp_gpu_cta() {
        let test = mp_gpu_scoped(Scope::CTA);
        assert_eq!(test.thread_count(), 2);
        assert!(test.name.contains("cta"));
    }

    #[test]
    fn test_mp_gpu_system() {
        let test = mp_gpu_scoped(Scope::System);
        assert!(test.name.contains("sys"));
    }

    #[test]
    fn test_mp_cross_scope() {
        let test = mp_cross_scope(Scope::CTA, Scope::GPU);
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_sb_gpu_scoped() {
        let test = sb_gpu_scoped(Scope::GPU);
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_lb_gpu_scoped() {
        let test = lb_gpu_scoped(Scope::CTA);
        assert_eq!(test.thread_count(), 2);
    }

    #[test]
    fn test_scope_to_release_mapping() {
        assert_eq!(scope_to_release(Scope::CTA), Ordering::ReleaseCTA);
        assert_eq!(scope_to_release(Scope::GPU), Ordering::ReleaseGPU);
        assert_eq!(scope_to_release(Scope::System), Ordering::ReleaseSystem);
        assert_eq!(scope_to_release(Scope::None), Ordering::Release);
    }

    #[test]
    fn test_scope_to_acquire_mapping() {
        assert_eq!(scope_to_acquire(Scope::CTA), Ordering::AcquireCTA);
        assert_eq!(scope_to_acquire(Scope::GPU), Ordering::AcquireGPU);
        assert_eq!(scope_to_acquire(Scope::System), Ordering::AcquireSystem);
        assert_eq!(scope_to_acquire(Scope::None), Ordering::Acquire);
    }

    #[test]
    fn test_catalog_all_tests_valid() {
        let cat = TestCatalog::standard();
        for entry in cat.entries() {
            assert!(!entry.name.is_empty());
            assert!(entry.test.thread_count() >= entry.pattern.min_threads());
        }
    }

    #[test]
    fn test_catalog_gpu_tests_have_tag() {
        let cat = TestCatalog::with_gpu();
        let gpu_tests = cat.by_tag("gpu");
        assert!(!gpu_tests.is_empty());
    }

    #[test]
    fn test_default_catalog() {
        let cat = TestCatalog::default();
        assert!(!cat.is_empty());
    }

    #[test]
    fn test_allowed_under() {
        let cat = TestCatalog::standard();
        let tso_allowed = cat.allowed_under("TSO");
        assert!(!tso_allowed.is_empty());
    }

    #[test]
    fn test_catalog_add() {
        let mut cat = TestCatalog::new();
        let entry = CatalogEntry::new("test", PatternKind::MP, mp_relaxed());
        cat.add(entry);
        assert_eq!(cat.len(), 1);
    }
}
