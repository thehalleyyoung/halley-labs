//! Coverage metrics for memory model testing in LITMUS∞.
//!
//! Provides axiom coverage, relation coverage, pattern coverage,
//! and coverage-guided test selection for memory model conformance testing.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use crate::checker::{
    LitmusTest, Thread, Instruction, Outcome, LitmusOutcome,
    Address, Value,
    MemoryModel, RelationExpr, Constraint, BuiltinModel,
    ExecutionGraph, BitMatrix,
};
use crate::checker::litmus::{Ordering, Scope, RegId};
use crate::checker::execution::OpType;

// ---------------------------------------------------------------------------
// CoverageMetric
// ---------------------------------------------------------------------------

/// A coverage metric for memory model testing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoverageMetric {
    /// Coverage of model axioms (acyclicity/irreflexivity constraints).
    Axiom(String),
    /// Coverage of a specific relation (rf, co, fr, po, etc.).
    Relation(String),
    /// Coverage of a structural pattern (MP, SB, LB, etc.).
    Pattern(String),
    /// Coverage of a specific ordering annotation.
    OrderingUsed(String),
    /// Coverage of a scope annotation.
    ScopeUsed(String),
    /// Coverage of instruction type.
    InstructionType(String),
    /// Coverage of thread count.
    ThreadCount(usize),
    /// Coverage of location count.
    LocationCount(usize),
    /// Cross-thread communication pattern.
    CommunicationPattern(String),
}

impl fmt::Display for CoverageMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Axiom(name) => write!(f, "axiom:{}", name),
            Self::Relation(name) => write!(f, "relation:{}", name),
            Self::Pattern(name) => write!(f, "pattern:{}", name),
            Self::OrderingUsed(name) => write!(f, "ordering:{}", name),
            Self::ScopeUsed(name) => write!(f, "scope:{}", name),
            Self::InstructionType(name) => write!(f, "instr-type:{}", name),
            Self::ThreadCount(n) => write!(f, "threads:{}", n),
            Self::LocationCount(n) => write!(f, "locations:{}", n),
            Self::CommunicationPattern(name) => write!(f, "comm:{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// CoverageConfig
// ---------------------------------------------------------------------------

/// Configuration for coverage tracking.
#[derive(Debug, Clone)]
pub struct CoverageConfig {
    /// Track axiom coverage.
    pub track_axioms: bool,
    /// Track relation coverage.
    pub track_relations: bool,
    /// Track pattern coverage.
    pub track_patterns: bool,
    /// Track ordering coverage.
    pub track_orderings: bool,
    /// Track scope coverage.
    pub track_scopes: bool,
    /// Track instruction type coverage.
    pub track_instr_types: bool,
    /// Track communication patterns.
    pub track_communication: bool,
    /// Memory model to use for axiom/relation coverage.
    pub model: Option<BuiltinModel>,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            track_axioms: true,
            track_relations: true,
            track_patterns: true,
            track_orderings: true,
            track_scopes: true,
            track_instr_types: true,
            track_communication: true,
            model: Some(BuiltinModel::SC),
        }
    }
}

impl CoverageConfig {
    /// Config for axiom-focused coverage.
    pub fn axiom_focused() -> Self {
        Self {
            track_axioms: true,
            track_relations: true,
            track_patterns: false,
            track_orderings: false,
            track_scopes: false,
            track_instr_types: false,
            track_communication: false,
            model: Some(BuiltinModel::SC),
        }
    }

    /// Config for pattern-focused coverage.
    pub fn pattern_focused() -> Self {
        Self {
            track_axioms: false,
            track_relations: false,
            track_patterns: true,
            track_orderings: true,
            track_scopes: false,
            track_instr_types: false,
            track_communication: true,
            model: None,
        }
    }
}

// ---------------------------------------------------------------------------
// AxiomCoverage
// ---------------------------------------------------------------------------

/// Tracks which axioms of a memory model are exercised by tests.
#[derive(Debug, Clone)]
pub struct AxiomCoverage {
    /// Model name.
    pub model_name: String,
    /// Axiom names from constraints.
    pub axiom_names: Vec<String>,
    /// Which axioms have been covered.
    pub covered: HashSet<String>,
    /// Tests that cover each axiom.
    pub covering_tests: HashMap<String, Vec<String>>,
}

impl AxiomCoverage {
    /// Create axiom coverage for a model.
    pub fn new(model: &MemoryModel) -> Self {
        let axiom_names: Vec<String> = model.constraints.iter()
            .map(|c| c.name().to_string())
            .collect();

        Self {
            model_name: model.name.clone(),
            axiom_names,
            covered: HashSet::new(),
            covering_tests: HashMap::new(),
        }
    }

    /// Create from a builtin model.
    pub fn from_builtin(builtin: BuiltinModel) -> Self {
        Self::new(&builtin.build())
    }

    /// Record that a test exercises an axiom.
    pub fn record(&mut self, axiom_name: &str, test_name: &str) {
        self.covered.insert(axiom_name.to_string());
        self.covering_tests
            .entry(axiom_name.to_string())
            .or_default()
            .push(test_name.to_string());
    }

    /// Check if an axiom is covered.
    pub fn is_covered(&self, axiom_name: &str) -> bool {
        self.covered.contains(axiom_name)
    }

    /// Total number of axioms.
    pub fn total(&self) -> usize {
        self.axiom_names.len()
    }

    /// Number of covered axioms.
    pub fn covered_count(&self) -> usize {
        self.covered.len()
    }

    /// Coverage ratio (0.0 to 1.0).
    pub fn ratio(&self) -> f64 {
        if self.total() == 0 { return 1.0; }
        self.covered_count() as f64 / self.total() as f64
    }

    /// Uncovered axioms.
    pub fn uncovered(&self) -> Vec<&str> {
        self.axiom_names.iter()
            .filter(|n| !self.covered.contains(n.as_str()))
            .map(|n| n.as_str())
            .collect()
    }

    /// Analyse a test against the model to determine which axioms it exercises.
    pub fn analyse_test(&mut self, test: &LitmusTest, model: &MemoryModel) {
        let executions = test.enumerate_executions();

        for (exec, _regs, _mem) in &executions {
            let env = model.compute_derived(exec);

            for constraint in &model.constraints {
                let rel = model.eval_expr(constraint.expr(), exec, &env);
                let is_exercised = match constraint {
                    Constraint::Acyclic { .. } => {
                        // Axiom is exercised if the relation has edges
                        // (a cycle would violate it)
                        !rel.is_empty()
                    }
                    Constraint::Irreflexive { .. } => {
                        // Exercised if there are diagonal entries to check
                        let n = rel.dim();
                        (0..n).any(|i| rel.get(i, i))
                            || !rel.is_empty()
                    }
                    Constraint::Empty { .. } => {
                        !rel.is_empty()
                    }
                };

                if is_exercised {
                    self.record(constraint.name(), &test.name);
                }
            }
        }
    }
}

impl fmt::Display for AxiomCoverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Axiom Coverage for {}: {}/{} ({:.1}%)",
            self.model_name,
            self.covered_count(),
            self.total(),
            self.ratio() * 100.0,
        )?;
        for name in &self.axiom_names {
            let status = if self.covered.contains(name) { "✓" } else { "✗" };
            let tests = self.covering_tests.get(name)
                .map(|t| t.len())
                .unwrap_or(0);
            writeln!(f, "  {} {} ({} tests)", status, name, tests)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RelationCoverage
// ---------------------------------------------------------------------------

/// Tracks coverage of relations (rf, co, fr, po, etc.).
#[derive(Debug, Clone)]
pub struct RelationCoverage {
    /// Relations being tracked.
    pub relations: Vec<String>,
    /// Which relations have been exercised (had non-empty instances).
    pub covered: HashSet<String>,
    /// Relation edge counts per test.
    pub edge_counts: HashMap<String, Vec<(String, usize)>>,
    /// Internal vs external usage.
    pub internal_used: HashSet<String>,
    pub external_used: HashSet<String>,
}

impl RelationCoverage {
    /// Create relation coverage for standard relations.
    pub fn standard() -> Self {
        Self {
            relations: vec![
                "po".into(), "rf".into(), "co".into(), "fr".into(),
                "rfe".into(), "rfi".into(), "coe".into(), "coi".into(),
                "fre".into(), "fri".into(), "po-loc".into(),
                "com".into(),
            ],
            covered: HashSet::new(),
            edge_counts: HashMap::new(),
            internal_used: HashSet::new(),
            external_used: HashSet::new(),
        }
    }

    /// Create with custom relation set.
    pub fn with_relations(names: Vec<String>) -> Self {
        Self {
            relations: names,
            covered: HashSet::new(),
            edge_counts: HashMap::new(),
            internal_used: HashSet::new(),
            external_used: HashSet::new(),
        }
    }

    /// Record relation usage from an execution.
    pub fn record_execution(&mut self, test_name: &str, exec: &ExecutionGraph) {
        let relations_to_check = vec![
            ("po", exec.po.clone()),
            ("rf", exec.rf.clone()),
            ("co", exec.co.clone()),
            ("fr", exec.fr.clone()),
        ];

        for (name, rel) in &relations_to_check {
            if !rel.is_empty() {
                self.covered.insert(name.to_string());
                self.edge_counts
                    .entry(name.to_string())
                    .or_default()
                    .push((test_name.to_string(), rel.count_edges()));

                // Check internal vs external.
                let int = exec.internal(rel);
                let ext = exec.external(rel);
                if !int.is_empty() {
                    self.internal_used.insert(name.to_string());
                }
                if !ext.is_empty() {
                    self.external_used.insert(name.to_string());
                }
            }
        }

        // Derived relations.
        let rfe = exec.external(&exec.rf);
        if !rfe.is_empty() {
            self.covered.insert("rfe".into());
        }
        let rfi = exec.internal(&exec.rf);
        if !rfi.is_empty() {
            self.covered.insert("rfi".into());
        }
        let coe = exec.external(&exec.co);
        if !coe.is_empty() {
            self.covered.insert("coe".into());
        }
        let coi = exec.internal(&exec.co);
        if !coi.is_empty() {
            self.covered.insert("coi".into());
        }
        let fre = exec.external(&exec.fr);
        if !fre.is_empty() {
            self.covered.insert("fre".into());
        }
        let fri = exec.internal(&exec.fr);
        if !fri.is_empty() {
            self.covered.insert("fri".into());
        }

        // po-loc
        let sa = exec.same_address();
        let po_loc = exec.po.intersection(&sa);
        if !po_loc.is_empty() {
            self.covered.insert("po-loc".into());
        }

        // com
        let com = exec.rf.union(&exec.co).union(&exec.fr);
        if !com.is_empty() {
            self.covered.insert("com".into());
        }
    }

    /// Record from a litmus test by enumerating executions.
    pub fn record_test(&mut self, test: &LitmusTest) {
        let executions = test.enumerate_executions();
        for (exec, _, _) in &executions {
            self.record_execution(&test.name, exec);
        }
    }

    /// Total number of tracked relations.
    pub fn total(&self) -> usize {
        self.relations.len()
    }

    /// Number of covered relations.
    pub fn covered_count(&self) -> usize {
        self.covered.intersection(&self.relations.iter().cloned().collect()).count()
    }

    /// Coverage ratio.
    pub fn ratio(&self) -> f64 {
        if self.total() == 0 { return 1.0; }
        self.covered_count() as f64 / self.total() as f64
    }

    /// Uncovered relations.
    pub fn uncovered(&self) -> Vec<&str> {
        self.relations.iter()
            .filter(|r| !self.covered.contains(r.as_str()))
            .map(|r| r.as_str())
            .collect()
    }
}

impl fmt::Display for RelationCoverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Relation Coverage: {}/{} ({:.1}%)",
            self.covered_count(),
            self.total(),
            self.ratio() * 100.0,
        )?;
        for name in &self.relations {
            let status = if self.covered.contains(name) { "✓" } else { "✗" };
            let edge_info = self.edge_counts.get(name)
                .map(|counts| {
                    let total: usize = counts.iter().map(|(_, c)| c).sum();
                    format!("{} edges across {} tests", total, counts.len())
                })
                .unwrap_or_else(|| "no edges".into());
            writeln!(f, "  {} {} ({})", status, name, edge_info)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PatternCoverage
// ---------------------------------------------------------------------------

/// Tracks coverage of structural litmus test patterns.
#[derive(Debug, Clone)]
pub struct PatternCoverage {
    /// Known patterns.
    pub patterns: Vec<String>,
    /// Covered patterns.
    pub covered: HashSet<String>,
    /// Tests covering each pattern.
    pub covering_tests: HashMap<String, Vec<String>>,
}

impl PatternCoverage {
    /// Create with standard pattern set.
    pub fn standard() -> Self {
        Self {
            patterns: vec![
                "MP".into(), "SB".into(), "LB".into(),
                "IRIW".into(), "WRC".into(), "ISA2".into(),
                "2+2W".into(), "R".into(), "S".into(),
                "RMW".into(),
            ],
            covered: HashSet::new(),
            covering_tests: HashMap::new(),
        }
    }

    /// Create with GPU patterns included.
    pub fn with_gpu() -> Self {
        Self {
            patterns: vec![
                "MP".into(), "SB".into(), "LB".into(),
                "IRIW".into(), "WRC".into(), "ISA2".into(),
                "2+2W".into(), "R".into(), "S".into(),
                "RMW".into(),
                "MP-cta".into(), "MP-gpu".into(), "MP-sys".into(),
                "SB-cta".into(), "SB-gpu".into(), "SB-sys".into(),
            ],
            covered: HashSet::new(),
            covering_tests: HashMap::new(),
        }
    }

    /// Record a pattern as covered.
    pub fn record(&mut self, pattern: &str, test_name: &str) {
        self.covered.insert(pattern.to_string());
        self.covering_tests
            .entry(pattern.to_string())
            .or_default()
            .push(test_name.to_string());
    }

    /// Classify a test and record its patterns.
    pub fn classify_and_record(&mut self, test: &LitmusTest) {
        let patterns = classify_test_pattern(test);
        for pat in patterns {
            self.record(&pat, &test.name);
        }
    }

    /// Total number of tracked patterns.
    pub fn total(&self) -> usize {
        self.patterns.len()
    }

    /// Number of covered patterns.
    pub fn covered_count(&self) -> usize {
        self.covered.intersection(&self.patterns.iter().cloned().collect()).count()
    }

    /// Coverage ratio.
    pub fn ratio(&self) -> f64 {
        if self.total() == 0 { return 1.0; }
        self.covered_count() as f64 / self.total() as f64
    }

    /// Uncovered patterns.
    pub fn uncovered(&self) -> Vec<&str> {
        self.patterns.iter()
            .filter(|p| !self.covered.contains(p.as_str()))
            .map(|p| p.as_str())
            .collect()
    }
}

impl fmt::Display for PatternCoverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pattern Coverage: {}/{} ({:.1}%)",
            self.covered_count(),
            self.total(),
            self.ratio() * 100.0,
        )?;
        for name in &self.patterns {
            let status = if self.covered.contains(name) { "✓" } else { "✗" };
            let tests = self.covering_tests.get(name)
                .map(|t| t.len())
                .unwrap_or(0);
            writeln!(f, "  {} {} ({} tests)", status, name, tests)?;
        }
        Ok(())
    }
}

/// Classify a litmus test into pattern categories.
fn classify_test_pattern(test: &LitmusTest) -> Vec<String> {
    let mut patterns = Vec::new();

    if test.thread_count() < 2 { return patterns; }

    let name_lower = test.name.to_lowercase();

    // Name-based classification.
    if name_lower.starts_with("mp") || name_lower.contains("message") {
        patterns.push("MP".into());
    }
    if name_lower.starts_with("sb") || name_lower.contains("store-buffer") {
        patterns.push("SB".into());
    }
    if name_lower.starts_with("lb") || name_lower.contains("load-buffer") {
        patterns.push("LB".into());
    }
    if name_lower.starts_with("iriw") {
        patterns.push("IRIW".into());
    }
    if name_lower.starts_with("wrc") {
        patterns.push("WRC".into());
    }
    if name_lower.starts_with("isa2") {
        patterns.push("ISA2".into());
    }
    if name_lower.starts_with("2+2w") || name_lower.contains("two_plus_two") {
        patterns.push("2+2W".into());
    }

    // Structural classification (if name didn't match).
    if patterns.is_empty() {
        patterns.extend(structural_classify(test));
    }

    // GPU scope classification.
    for t in &test.threads {
        for instr in &t.instructions {
            match instr {
                Instruction::Fence { scope: Scope::CTA, .. } => {
                    if !patterns.iter().any(|p| p.ends_with("-cta")) {
                        for p in patterns.clone() {
                            patterns.push(format!("{}-cta", p));
                        }
                    }
                }
                Instruction::Fence { scope: Scope::GPU, .. } => {
                    if !patterns.iter().any(|p| p.ends_with("-gpu")) {
                        for p in patterns.clone() {
                            patterns.push(format!("{}-gpu", p));
                        }
                    }
                }
                Instruction::Fence { scope: Scope::System, .. } => {
                    if !patterns.iter().any(|p| p.ends_with("-sys")) {
                        for p in patterns.clone() {
                            patterns.push(format!("{}-sys", p));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // RMW patterns.
    let has_rmw = test.threads.iter().any(|t|
        t.instructions.iter().any(|i| matches!(i, Instruction::RMW { .. }))
    );
    if has_rmw {
        patterns.push("RMW".into());
    }

    patterns
}

/// Structural classification based on instruction patterns.
fn structural_classify(test: &LitmusTest) -> Vec<String> {
    let mut patterns = Vec::new();

    if test.thread_count() < 2 { return patterns; }

    let t0 = &test.threads[0];
    let t1 = &test.threads[1];

    // MP: T0 writes, T1 reads.
    let t0_all_writes = t0.instructions.iter().all(|i|
        matches!(i, Instruction::Store { .. } | Instruction::Fence { .. })
    );
    let t1_all_reads = t1.instructions.iter().all(|i|
        matches!(i, Instruction::Load { .. } | Instruction::Fence { .. })
    );
    if t0_all_writes && t1_all_reads {
        patterns.push("MP".into());
    }

    // SB: Both threads store then load.
    let t0_store_first = t0.instructions.first().map_or(false, |i| matches!(i, Instruction::Store { .. }));
    let t0_load_last = t0.instructions.last().map_or(false, |i| matches!(i, Instruction::Load { .. }));
    let t1_store_first = t1.instructions.first().map_or(false, |i| matches!(i, Instruction::Store { .. }));
    let t1_load_last = t1.instructions.last().map_or(false, |i| matches!(i, Instruction::Load { .. }));
    if t0_store_first && t0_load_last && t1_store_first && t1_load_last {
        patterns.push("SB".into());
    }

    // LB: Both threads load then store.
    let t0_load_first = t0.instructions.first().map_or(false, |i| matches!(i, Instruction::Load { .. }));
    let t0_store_last = t0.instructions.last().map_or(false, |i| matches!(i, Instruction::Store { .. }));
    let t1_load_first = t1.instructions.first().map_or(false, |i| matches!(i, Instruction::Load { .. }));
    let t1_store_last = t1.instructions.last().map_or(false, |i| matches!(i, Instruction::Store { .. }));
    if t0_load_first && t0_store_last && t1_load_first && t1_store_last {
        patterns.push("LB".into());
    }

    // IRIW: 4+ threads.
    if test.thread_count() >= 4 {
        patterns.push("IRIW".into());
    }

    // WRC / ISA2: 3+ threads.
    if test.thread_count() >= 3 {
        let writes_first_thread = t0.instructions.iter()
            .filter(|i| matches!(i, Instruction::Store { .. }))
            .count();
        if writes_first_thread >= 1 {
            patterns.push("WRC".into());
        }
    }

    // R pattern: same location, two writes then two reads.
    let t0_writes_same_loc = t0.instructions.iter()
        .filter_map(|i| if let Instruction::Store { addr, .. } = i { Some(addr) } else { None })
        .collect::<HashSet<_>>();
    if t0_writes_same_loc.len() == 1 && t0.instructions.iter()
        .filter(|i| matches!(i, Instruction::Store { .. }))
        .count() >= 2
    {
        patterns.push("R".into());
    }

    // S pattern: writes from different threads to same location.
    let mut write_addrs: HashMap<Address, HashSet<usize>> = HashMap::new();
    for t in &test.threads {
        for instr in &t.instructions {
            if let Instruction::Store { addr, .. } = instr {
                write_addrs.entry(*addr).or_default().insert(t.id);
            }
        }
    }
    if write_addrs.values().any(|threads| threads.len() >= 2) {
        patterns.push("S".into());
    }

    patterns
}

// ---------------------------------------------------------------------------
// CoverageTracker — composite tracker
// ---------------------------------------------------------------------------

/// Composite coverage tracker that combines multiple metrics.
#[derive(Debug, Clone)]
pub struct CoverageTracker {
    pub config: CoverageConfig,
    pub axiom_coverage: Option<AxiomCoverage>,
    pub relation_coverage: RelationCoverage,
    pub pattern_coverage: PatternCoverage,
    /// Ordering annotations seen.
    pub orderings_seen: HashSet<String>,
    /// Scopes seen.
    pub scopes_seen: HashSet<String>,
    /// Instruction types seen.
    pub instr_types_seen: HashSet<String>,
    /// Thread counts seen.
    pub thread_counts_seen: HashSet<usize>,
    /// Location counts seen.
    pub location_counts_seen: HashSet<usize>,
    /// Total tests analysed.
    pub tests_analysed: usize,
}

impl CoverageTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: CoverageConfig) -> Self {
        let axiom_coverage = config.model.map(|m| AxiomCoverage::from_builtin(m));
        Self {
            config,
            axiom_coverage,
            relation_coverage: RelationCoverage::standard(),
            pattern_coverage: PatternCoverage::standard(),
            orderings_seen: HashSet::new(),
            scopes_seen: HashSet::new(),
            instr_types_seen: HashSet::new(),
            thread_counts_seen: HashSet::new(),
            location_counts_seen: HashSet::new(),
            tests_analysed: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_tracker() -> Self {
        Self::new(CoverageConfig::default())
    }

    /// Record a test.
    pub fn record_test(&mut self, test: &LitmusTest) {
        self.tests_analysed += 1;

        // Pattern coverage.
        if self.config.track_patterns {
            self.pattern_coverage.classify_and_record(test);
        }

        // Relation coverage.
        if self.config.track_relations {
            self.relation_coverage.record_test(test);
        }

        // Axiom coverage.
        if self.config.track_axioms {
            if let Some(builtin) = self.config.model {
                let model = builtin.build();
                if let Some(ref mut cov) = self.axiom_coverage {
                    cov.analyse_test(test, &model);
                }
            }
        }

        // Ordering coverage.
        if self.config.track_orderings {
            for t in &test.threads {
                for instr in &t.instructions {
                    let ord = match instr {
                        Instruction::Load { ordering, .. } => Some(*ordering),
                        Instruction::Store { ordering, .. } => Some(*ordering),
                        Instruction::Fence { ordering, .. } => Some(*ordering),
                        Instruction::RMW { ordering, .. } => Some(*ordering),
                        _ => None,
                    };
                    if let Some(o) = ord {
                        self.orderings_seen.insert(format!("{}", o));
                    }
                }
            }
        }

        // Scope coverage.
        if self.config.track_scopes {
            for t in &test.threads {
                for instr in &t.instructions {
                    if let Instruction::Fence { scope, .. } = instr {
                        self.scopes_seen.insert(format!("{}", scope));
                    }
                }
            }
        }

        // Instruction type coverage.
        if self.config.track_instr_types {
            for t in &test.threads {
                for instr in &t.instructions {
                    let ty = match instr {
                        Instruction::Load { .. } => "load",
                        Instruction::Store { .. } => "store",
                        Instruction::Fence { .. } => "fence",
                        Instruction::RMW { .. } => "rmw",
                        Instruction::Branch { .. } => "branch",
                        Instruction::Label { .. } => "label",
                        Instruction::BranchCond { .. } => "branch-cond",
                    };
                    self.instr_types_seen.insert(ty.into());
                }
            }
        }

        // Thread count.
        self.thread_counts_seen.insert(test.thread_count());

        // Location count.
        self.location_counts_seen.insert(test.all_addresses().len());
    }

    /// Record multiple tests.
    pub fn record_tests(&mut self, tests: &[LitmusTest]) {
        for test in tests {
            self.record_test(test);
        }
    }

    /// Generate a coverage report.
    pub fn report(&self) -> CoverageReport {
        let mut metrics = Vec::new();

        // Axiom metrics.
        if let Some(ref cov) = self.axiom_coverage {
            for name in &cov.axiom_names {
                metrics.push(CoverageMetric::Axiom(name.clone()));
            }
        }

        // Relation metrics.
        for name in &self.relation_coverage.relations {
            metrics.push(CoverageMetric::Relation(name.clone()));
        }

        // Pattern metrics.
        for name in &self.pattern_coverage.patterns {
            metrics.push(CoverageMetric::Pattern(name.clone()));
        }

        let mut covered_metrics = HashSet::new();
        if let Some(ref cov) = self.axiom_coverage {
            for name in &cov.covered {
                covered_metrics.insert(CoverageMetric::Axiom(name.clone()));
            }
        }
        for name in &self.relation_coverage.covered {
            covered_metrics.insert(CoverageMetric::Relation(name.clone()));
        }
        for name in &self.pattern_coverage.covered {
            covered_metrics.insert(CoverageMetric::Pattern(name.clone()));
        }
        for ord in &self.orderings_seen {
            covered_metrics.insert(CoverageMetric::OrderingUsed(ord.clone()));
        }
        for scope in &self.scopes_seen {
            covered_metrics.insert(CoverageMetric::ScopeUsed(scope.clone()));
        }
        for ty in &self.instr_types_seen {
            covered_metrics.insert(CoverageMetric::InstructionType(ty.clone()));
        }
        for &n in &self.thread_counts_seen {
            covered_metrics.insert(CoverageMetric::ThreadCount(n));
        }
        for &n in &self.location_counts_seen {
            covered_metrics.insert(CoverageMetric::LocationCount(n));
        }

        CoverageReport {
            total_metrics: metrics.len(),
            covered_metrics: covered_metrics.len(),
            metrics,
            covered: covered_metrics,
            tests_analysed: self.tests_analysed,
        }
    }

    /// Overall coverage ratio.
    pub fn overall_ratio(&self) -> f64 {
        let report = self.report();
        if report.total_metrics == 0 { return 1.0; }
        report.covered_metrics as f64 / report.total_metrics as f64
    }
}

impl fmt::Display for CoverageTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Coverage Report ({} tests analysed) ===", self.tests_analysed)?;
        writeln!(f)?;

        if let Some(ref cov) = self.axiom_coverage {
            write!(f, "{}", cov)?;
            writeln!(f)?;
        }

        write!(f, "{}", self.relation_coverage)?;
        writeln!(f)?;

        write!(f, "{}", self.pattern_coverage)?;
        writeln!(f)?;

        writeln!(f, "Orderings used: {:?}", self.orderings_seen)?;
        writeln!(f, "Scopes used: {:?}", self.scopes_seen)?;
        writeln!(f, "Instruction types: {:?}", self.instr_types_seen)?;
        writeln!(f, "Thread counts: {:?}", self.thread_counts_seen)?;
        writeln!(f, "Location counts: {:?}", self.location_counts_seen)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CoverageReport
// ---------------------------------------------------------------------------

/// A coverage report summarising all metrics.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub total_metrics: usize,
    pub covered_metrics: usize,
    pub metrics: Vec<CoverageMetric>,
    pub covered: HashSet<CoverageMetric>,
    pub tests_analysed: usize,
}

impl CoverageReport {
    /// Coverage ratio.
    pub fn ratio(&self) -> f64 {
        if self.total_metrics == 0 { return 1.0; }
        self.covered_metrics as f64 / self.total_metrics as f64
    }

    /// Uncovered metrics.
    pub fn uncovered(&self) -> Vec<&CoverageMetric> {
        self.metrics.iter().filter(|m| !self.covered.contains(m)).collect()
    }

    /// Covered metrics.
    pub fn covered_list(&self) -> Vec<&CoverageMetric> {
        self.metrics.iter().filter(|m| self.covered.contains(m)).collect()
    }
}

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coverage: {}/{} metrics ({:.1}%), {} tests analysed",
            self.covered_metrics, self.total_metrics,
            self.ratio() * 100.0, self.tests_analysed)?;

        let uncov = self.uncovered();
        if !uncov.is_empty() {
            writeln!(f, "Uncovered:")?;
            for m in uncov {
                writeln!(f, "  - {}", m)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Coverage-guided selection
// ---------------------------------------------------------------------------

/// Select a subset of tests that maximises coverage.
pub fn greedy_coverage_selection(
    tests: &[LitmusTest],
    config: &CoverageConfig,
    max_tests: usize,
) -> Vec<usize> {
    let mut tracker = CoverageTracker::new(config.clone());
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: HashSet<usize> = (0..tests.len()).collect();

    while selected.len() < max_tests && !remaining.is_empty() {
        let mut best_idx = None;
        let mut best_gain = 0usize;

        for &idx in &remaining {
            // Simulate adding this test.
            let mut tmp = tracker.clone();
            tmp.record_test(&tests[idx]);
            let gain = tmp.report().covered_metrics.saturating_sub(tracker.report().covered_metrics);
            if gain > best_gain || (best_idx.is_none() && gain == 0) {
                best_gain = gain;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            tracker.record_test(&tests[idx]);
            selected.push(idx);
            remaining.remove(&idx);

            // Stop if no coverage gain.
            if best_gain == 0 && selected.len() > 1 {
                break;
            }
        } else {
            break;
        }
    }

    selected
}

/// Compute coverage delta: what new metrics does a test add?
pub fn coverage_delta(
    test: &LitmusTest,
    tracker: &CoverageTracker,
) -> Vec<CoverageMetric> {
    let mut tmp = tracker.clone();
    tmp.record_test(test);

    let old_report = tracker.report();
    let new_report = tmp.report();

    new_report.covered.difference(&old_report.covered).cloned().collect()
}

/// Compute redundancy: how many tests cover the same metrics?
pub fn compute_redundancy(
    tests: &[LitmusTest],
    config: &CoverageConfig,
) -> HashMap<CoverageMetric, usize> {
    let mut counts: HashMap<CoverageMetric, usize> = HashMap::new();

    for test in tests {
        let mut tracker = CoverageTracker::new(config.clone());
        tracker.record_test(test);
        let report = tracker.report();
        for m in &report.covered {
            *counts.entry(m.clone()).or_default() += 1;
        }
    }

    counts
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mp_test() -> LitmusTest {
        crate::checker::litmus::mp_test()
    }

    fn make_sb_test() -> LitmusTest {
        crate::checker::litmus::sb_test()
    }

    fn make_lb_test() -> LitmusTest {
        crate::checker::litmus::lb_test()
    }

    #[test]
    fn test_coverage_metric_display() {
        let m = CoverageMetric::Axiom("acyclic".into());
        assert_eq!(format!("{}", m), "axiom:acyclic");

        let m = CoverageMetric::Relation("rf".into());
        assert_eq!(format!("{}", m), "relation:rf");

        let m = CoverageMetric::Pattern("MP".into());
        assert_eq!(format!("{}", m), "pattern:MP");
    }

    #[test]
    fn test_coverage_config_default() {
        let config = CoverageConfig::default();
        assert!(config.track_axioms);
        assert!(config.track_relations);
        assert!(config.track_patterns);
    }

    #[test]
    fn test_coverage_config_axiom_focused() {
        let config = CoverageConfig::axiom_focused();
        assert!(config.track_axioms);
        assert!(!config.track_patterns);
    }

    #[test]
    fn test_coverage_config_pattern_focused() {
        let config = CoverageConfig::pattern_focused();
        assert!(!config.track_axioms);
        assert!(config.track_patterns);
    }

    #[test]
    fn test_axiom_coverage_sc() {
        let model = BuiltinModel::SC.build();
        let cov = AxiomCoverage::new(&model);
        assert!(cov.total() > 0);
        assert_eq!(cov.covered_count(), 0);
    }

    #[test]
    fn test_axiom_coverage_record() {
        let model = BuiltinModel::SC.build();
        let mut cov = AxiomCoverage::new(&model);
        let axiom = &model.constraints[0].name().to_string();
        cov.record(axiom, "test-1");
        assert!(cov.is_covered(axiom));
        assert_eq!(cov.covered_count(), 1);
    }

    #[test]
    fn test_axiom_coverage_ratio() {
        let model = BuiltinModel::SC.build();
        let mut cov = AxiomCoverage::new(&model);
        assert_eq!(cov.ratio(), 0.0);
        for c in &model.constraints {
            cov.record(c.name(), "test");
        }
        assert!((cov.ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_axiom_coverage_uncovered() {
        let model = BuiltinModel::SC.build();
        let cov = AxiomCoverage::new(&model);
        let uncov = cov.uncovered();
        assert_eq!(uncov.len(), cov.total());
    }

    #[test]
    fn test_axiom_coverage_display() {
        let cov = AxiomCoverage::from_builtin(BuiltinModel::SC);
        let s = format!("{}", cov);
        assert!(s.contains("Axiom Coverage"));
    }

    #[test]
    fn test_relation_coverage_standard() {
        let cov = RelationCoverage::standard();
        assert!(cov.total() > 0);
        assert_eq!(cov.covered_count(), 0);
    }

    #[test]
    fn test_relation_coverage_record_test() {
        let mut cov = RelationCoverage::standard();
        let test = make_mp_test();
        cov.record_test(&test);
        // po should always be covered for multi-thread tests with instructions.
        assert!(cov.covered.contains("po"));
    }

    #[test]
    fn test_relation_coverage_ratio_initially_zero() {
        let cov = RelationCoverage::standard();
        assert_eq!(cov.ratio(), 0.0);
    }

    #[test]
    fn test_relation_coverage_uncovered() {
        let cov = RelationCoverage::standard();
        assert_eq!(cov.uncovered().len(), cov.total());
    }

    #[test]
    fn test_relation_coverage_display() {
        let cov = RelationCoverage::standard();
        let s = format!("{}", cov);
        assert!(s.contains("Relation Coverage"));
    }

    #[test]
    fn test_pattern_coverage_standard() {
        let cov = PatternCoverage::standard();
        assert!(cov.total() > 0);
    }

    #[test]
    fn test_pattern_coverage_with_gpu() {
        let cov = PatternCoverage::with_gpu();
        assert!(cov.total() > PatternCoverage::standard().total());
    }

    #[test]
    fn test_pattern_coverage_record() {
        let mut cov = PatternCoverage::standard();
        cov.record("MP", "MP-test");
        assert_eq!(cov.covered_count(), 1);
    }

    #[test]
    fn test_pattern_coverage_classify_mp() {
        let mut cov = PatternCoverage::standard();
        let test = make_mp_test();
        cov.classify_and_record(&test);
        assert!(cov.covered.contains("MP"));
    }

    #[test]
    fn test_pattern_coverage_classify_sb() {
        let mut cov = PatternCoverage::standard();
        let test = make_sb_test();
        cov.classify_and_record(&test);
        assert!(cov.covered.contains("SB"));
    }

    #[test]
    fn test_pattern_coverage_classify_lb() {
        let mut cov = PatternCoverage::standard();
        let test = make_lb_test();
        cov.classify_and_record(&test);
        assert!(cov.covered.contains("LB"));
    }

    #[test]
    fn test_classify_test_pattern_mp() {
        let test = make_mp_test();
        let patterns = classify_test_pattern(&test);
        assert!(patterns.contains(&"MP".to_string()));
    }

    #[test]
    fn test_classify_test_pattern_sb() {
        let test = make_sb_test();
        let patterns = classify_test_pattern(&test);
        assert!(patterns.contains(&"SB".to_string()));
    }

    #[test]
    fn test_classify_test_pattern_lb() {
        let test = make_lb_test();
        let patterns = classify_test_pattern(&test);
        assert!(patterns.contains(&"LB".to_string()));
    }

    #[test]
    fn test_classify_single_thread() {
        let mut test = LitmusTest::new("single");
        test.add_thread(Thread::new(0));
        let patterns = classify_test_pattern(&test);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_structural_classify_mp() {
        let test = make_mp_test();
        let patterns = structural_classify(&test);
        assert!(patterns.contains(&"MP".to_string()));
    }

    #[test]
    fn test_structural_classify_sb() {
        let test = make_sb_test();
        let patterns = structural_classify(&test);
        assert!(patterns.contains(&"SB".to_string()));
    }

    #[test]
    fn test_structural_classify_iriw() {
        let test = crate::checker::litmus::iriw_test();
        let patterns = structural_classify(&test);
        assert!(patterns.contains(&"IRIW".to_string()));
    }

    #[test]
    fn test_coverage_tracker_new() {
        let tracker = CoverageTracker::default_tracker();
        assert_eq!(tracker.tests_analysed, 0);
    }

    #[test]
    fn test_coverage_tracker_record_test() {
        let mut tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let test = make_mp_test();
        tracker.record_test(&test);
        assert_eq!(tracker.tests_analysed, 1);
    }

    #[test]
    fn test_coverage_tracker_record_multiple() {
        let mut tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let tests = vec![make_mp_test(), make_sb_test(), make_lb_test()];
        tracker.record_tests(&tests);
        assert_eq!(tracker.tests_analysed, 3);
    }

    #[test]
    fn test_coverage_tracker_orderings() {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        let test = make_mp_test();
        tracker.record_test(&test);
        assert!(tracker.orderings_seen.contains("rlx"));
    }

    #[test]
    fn test_coverage_tracker_thread_counts() {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        let test = make_mp_test();
        tracker.record_test(&test);
        assert!(tracker.thread_counts_seen.contains(&2));
    }

    #[test]
    fn test_coverage_tracker_location_counts() {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        let test = make_mp_test();
        tracker.record_test(&test);
        assert!(tracker.location_counts_seen.len() > 0);
    }

    #[test]
    fn test_coverage_tracker_report() {
        let mut tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let test = make_mp_test();
        tracker.record_test(&test);
        let report = tracker.report();
        assert!(report.total_metrics > 0);
        assert!(report.covered_metrics > 0);
    }

    #[test]
    fn test_coverage_tracker_display() {
        let mut tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let test = make_mp_test();
        tracker.record_test(&test);
        let s = format!("{}", tracker);
        assert!(s.contains("Coverage Report"));
    }

    #[test]
    fn test_coverage_report_ratio() {
        let report = CoverageReport {
            total_metrics: 10,
            covered_metrics: 5,
            metrics: vec![],
            covered: HashSet::new(),
            tests_analysed: 3,
        };
        assert!((report.ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_report_empty() {
        let report = CoverageReport {
            total_metrics: 0,
            covered_metrics: 0,
            metrics: vec![],
            covered: HashSet::new(),
            tests_analysed: 0,
        };
        assert!((report.ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_report_display() {
        let report = CoverageReport {
            total_metrics: 10,
            covered_metrics: 5,
            metrics: vec![CoverageMetric::Pattern("MP".into())],
            covered: HashSet::new(),
            tests_analysed: 3,
        };
        let s = format!("{}", report);
        assert!(s.contains("Coverage:"));
    }

    #[test]
    fn test_greedy_selection() {
        let tests = vec![make_mp_test(), make_sb_test(), make_lb_test()];
        let config = CoverageConfig::pattern_focused();
        let selected = greedy_coverage_selection(&tests, &config, 2);
        assert!(!selected.is_empty());
        assert!(selected.len() <= 2);
    }

    #[test]
    fn test_greedy_selection_all() {
        let tests = vec![make_mp_test(), make_sb_test()];
        let config = CoverageConfig::pattern_focused();
        let selected = greedy_coverage_selection(&tests, &config, 10);
        assert!(selected.len() <= tests.len());
    }

    #[test]
    fn test_coverage_delta() {
        let tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let test = make_mp_test();
        let delta = coverage_delta(&test, &tracker);
        assert!(!delta.is_empty());
    }

    #[test]
    fn test_coverage_delta_after_recording() {
        let mut tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let test = make_mp_test();
        tracker.record_test(&test);
        // Same test again should have no new coverage.
        let delta = coverage_delta(&test, &tracker);
        assert!(delta.is_empty());
    }

    #[test]
    fn test_compute_redundancy() {
        let tests = vec![make_mp_test(), make_mp_test(), make_sb_test()];
        let config = CoverageConfig::pattern_focused();
        let redundancy = compute_redundancy(&tests, &config);
        // MP pattern should have count >= 2
        let mp_count = redundancy.get(&CoverageMetric::Pattern("MP".into()));
        assert!(mp_count.map_or(false, |&c| c >= 2));
    }

    #[test]
    fn test_overall_ratio_empty() {
        let tracker = CoverageTracker::new(CoverageConfig::pattern_focused());
        let ratio = tracker.overall_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_instr_types_coverage() {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        let test = make_mp_test();
        tracker.record_test(&test);
        assert!(tracker.instr_types_seen.contains("load"));
        assert!(tracker.instr_types_seen.contains("store"));
    }

    #[test]
    fn test_pattern_coverage_ratio() {
        let mut cov = PatternCoverage::standard();
        cov.record("MP", "test");
        cov.record("SB", "test");
        let ratio = cov.ratio();
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[test]
    fn test_pattern_coverage_uncovered() {
        let mut cov = PatternCoverage::standard();
        cov.record("MP", "test");
        let uncov = cov.uncovered();
        assert!(!uncov.contains(&"MP"));
        assert!(uncov.len() < cov.total());
    }

    #[test]
    fn test_relation_coverage_with_relations() {
        let cov = RelationCoverage::with_relations(vec!["rf".into(), "co".into()]);
        assert_eq!(cov.total(), 2);
    }

    #[test]
    fn test_classify_rmw_pattern() {
        let mut test = LitmusTest::new("rmw-test");
        let mut t0 = Thread::new(0);
        t0.rmw(0, 0x100, 1, Ordering::SeqCst);
        test.add_thread(t0);
        test.add_thread(Thread::new(1));
        let patterns = classify_test_pattern(&test);
        assert!(patterns.contains(&"RMW".to_string()));
    }
}
