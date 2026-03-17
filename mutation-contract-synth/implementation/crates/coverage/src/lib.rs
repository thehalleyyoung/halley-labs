//! # coverage
//!
//! Coverage analysis, mutant subsumption, dominator set computation, and scoring
//! for the MutSpec mutation-contract-synth project.
//!
//! This crate implements:
//! - **Subsumption analysis**: determines when one mutant subsumes another
//! - **Dominator set computation**: finds the minimal set of mutants that
//!   generates the same specification as the full kill set (Theorem T2)
//! - **Scoring**: various mutation score metrics
//! - **Equivalence detection**: identifies equivalent mutants via TCE, SMT,
//!   and heuristic methods
//! - **Adequacy analysis**: checks if a test suite is mutation-adequate
//! - **Metrics**: aggregated coverage metrics and trend analysis
//! - **Filtering**: mutant filtering pipelines
//!
//! ## Architecture
//!
//! The coverage crate sits between `mutation-core` (which generates mutants and
//! runs tests) and `contract-synth` (which synthesizes specifications from
//! surviving mutants). The key data structures—[`SubsumptionGraph`] and
//! [`DominatorSet`]—flow into the contract synthesis pipeline.

pub mod adequacy;
pub mod dominator;
pub mod equivalence;
pub mod filtering;
pub mod metrics;
pub mod scoring;
pub mod subsumption;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use adequacy::{AdequacyAnalyzer, AdequacyCertificate, AdequacyGap};
pub use dominator::{DominatorQuality, DominatorSet, DominatorSetComputer, DominatorStats};
pub use equivalence::{EquivalenceDetector, EquivalenceResult, EquivalenceStats};
pub use filtering::{FilterKind, FilterResult, MutantFilterPipeline};
pub use metrics::{CoverageMetrics, MetricAlert, MetricDelta, MetricThreshold};
pub use scoring::{FunctionScore, MutationScore, OperatorScore, ScoreBreakdown, TestEffectiveness};
pub use subsumption::{SubsumptionAnalyzer, SubsumptionEdge, SubsumptionGraph, SubsumptionStats};

// ── Local type definitions ──────────────────────────────────────────────────
//
// These mirror the types declared in `shared-types`, `mutation-core`, and
// `smt-solver`. Once those crates are fully implemented, these definitions
// should be replaced with re-exports from the upstream crates.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// Identifiers
// ────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a mutant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MutantId(pub String);

impl MutantId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MutantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for MutantId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for MutantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Unique identifier for a test case.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TestId(pub String);

impl TestId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for TestId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for TestId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Source locations
// ────────────────────────────────────────────────────────────────────────────

/// A location in source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A span in source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanInfo {
    pub start: SourceLocation,
    pub end: SourceLocation,
}

// ────────────────────────────────────────────────────────────────────────────
// Mutation operators
// ────────────────────────────────────────────────────────────────────────────

/// Mutation operator kinds used in MutSpec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MutationOperator {
    /// Arithmetic Operator Replacement (e.g., `+` → `-`)
    AOR,
    /// Relational Operator Replacement (e.g., `<` → `<=`)
    ROR,
    /// Conditional Operator Replacement (e.g., `&&` → `||`)
    COR,
    /// Shift Operator Replacement
    SOR,
    /// Logical Connector Replacement
    LCR,
    /// Unary Operator Insertion (e.g., negate a variable)
    UOI,
    /// Absolute Value Insertion
    ABS,
    /// Statement Deletion
    SDL,
    /// Constant Replacement
    CR,
    /// Variable Replacement
    VR,
    /// Return Value Replacement
    RVR,
    /// Assignment Operator Replacement
    ASGR,
    /// Bomb statement insertion (always fail)
    BOMB,
}

impl MutationOperator {
    /// Returns all known operators.
    pub fn all() -> &'static [MutationOperator] {
        use MutationOperator::*;
        &[
            AOR, ROR, COR, SOR, LCR, UOI, ABS, SDL, CR, VR, RVR, ASGR, BOMB,
        ]
    }

    /// Short name for display.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::AOR => "AOR",
            Self::ROR => "ROR",
            Self::COR => "COR",
            Self::SOR => "SOR",
            Self::LCR => "LCR",
            Self::UOI => "UOI",
            Self::ABS => "ABS",
            Self::SDL => "SDL",
            Self::CR => "CR",
            Self::VR => "VR",
            Self::RVR => "RVR",
            Self::ASGR => "ASGR",
            Self::BOMB => "BOMB",
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::AOR => "Arithmetic Operator Replacement",
            Self::ROR => "Relational Operator Replacement",
            Self::COR => "Conditional Operator Replacement",
            Self::SOR => "Shift Operator Replacement",
            Self::LCR => "Logical Connector Replacement",
            Self::UOI => "Unary Operator Insertion",
            Self::ABS => "Absolute Value Insertion",
            Self::SDL => "Statement Deletion",
            Self::CR => "Constant Replacement",
            Self::VR => "Variable Replacement",
            Self::RVR => "Return Value Replacement",
            Self::ASGR => "Assignment Operator Replacement",
            Self::BOMB => "Bomb Statement Insertion",
        }
    }
}

impl fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Mutation sites & descriptors
// ────────────────────────────────────────────────────────────────────────────

/// Describes where in the program a mutation was applied.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutationSite {
    pub file: String,
    pub function_name: String,
    pub line: usize,
    pub column: usize,
    pub span: Option<SpanInfo>,
}

impl MutationSite {
    pub fn new(
        file: impl Into<String>,
        function_name: impl Into<String>,
        line: usize,
        column: usize,
    ) -> Self {
        Self {
            file: file.into(),
            function_name: function_name.into(),
            line,
            column,
            span: None,
        }
    }

    pub fn with_span(mut self, span: SpanInfo) -> Self {
        self.span = Some(span);
        self
    }
}

impl fmt::Display for MutationSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{} ({})",
            self.file, self.line, self.column, self.function_name
        )
    }
}

/// Full description of a mutant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutantDescriptor {
    pub id: MutantId,
    pub operator: MutationOperator,
    pub site: MutationSite,
    pub original: String,
    pub replacement: String,
    pub description: String,
}

impl MutantDescriptor {
    pub fn new(
        id: MutantId,
        operator: MutationOperator,
        site: MutationSite,
        original: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        let original = original.into();
        let replacement = replacement.into();
        let description = format!("{}: {} → {} at {}", operator, original, replacement, site);
        Self {
            id,
            operator,
            site,
            original,
            replacement,
            description,
        }
    }
}

impl fmt::Display for MutantDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.id, self.description)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Mutant status
// ────────────────────────────────────────────────────────────────────────────

/// Information about how a mutant was killed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KillInfo {
    pub killing_test: TestId,
    pub assertion_message: Option<String>,
    pub expected_output: Option<String>,
    pub actual_output: Option<String>,
}

impl KillInfo {
    pub fn new(killing_test: TestId) -> Self {
        Self {
            killing_test,
            assertion_message: None,
            expected_output: None,
            actual_output: None,
        }
    }

    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.assertion_message = Some(msg.into());
        self
    }
}

/// Status of a mutant after test execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutantStatus {
    /// Not yet tested.
    Pending,
    /// Alive: survived all tests.
    Alive,
    /// Killed by at least one test.
    Killed(KillInfo),
    /// Provably equivalent to the original program.
    Equivalent,
    /// Test execution timed out.
    Timeout,
    /// Error during mutant compilation or test execution.
    Error(String),
}

impl MutantStatus {
    pub fn is_killed(&self) -> bool {
        matches!(self, Self::Killed(_))
    }

    pub fn is_alive(&self) -> bool {
        matches!(self, Self::Alive)
    }

    pub fn is_equivalent(&self) -> bool {
        matches!(self, Self::Equivalent)
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
}

impl fmt::Display for MutantStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "PENDING"),
            Self::Alive => write!(f, "ALIVE"),
            Self::Killed(info) => write!(f, "KILLED by {}", info.killing_test),
            Self::Equivalent => write!(f, "EQUIVALENT"),
            Self::Timeout => write!(f, "TIMEOUT"),
            Self::Error(msg) => write!(f, "ERROR: {}", msg),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Mutant (from mutation-core)
// ────────────────────────────────────────────────────────────────────────────

/// A concrete mutant with its descriptor and current status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutant {
    pub descriptor: MutantDescriptor,
    pub status: MutantStatus,
}

impl Mutant {
    pub fn new(descriptor: MutantDescriptor) -> Self {
        Self {
            descriptor,
            status: MutantStatus::Pending,
        }
    }

    pub fn with_status(mut self, status: MutantStatus) -> Self {
        self.status = status;
        self
    }

    pub fn id(&self) -> &MutantId {
        &self.descriptor.id
    }

    pub fn operator(&self) -> MutationOperator {
        self.descriptor.operator
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Kill matrix (from mutation-core)
// ────────────────────────────────────────────────────────────────────────────

/// Outcome of running a single test against a single mutant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    /// The test killed the mutant (detected the mutation).
    Killed,
    /// The test did not kill the mutant.
    Survived,
    /// The test timed out.
    Timeout,
    /// An error occurred.
    Error,
    /// Not yet run.
    NotRun,
}

impl TestOutcome {
    pub fn is_kill(&self) -> bool {
        matches!(self, Self::Killed)
    }
}

/// Matrix recording test × mutant outcomes.
///
/// Entry `(t, m)` records whether test `t` killed mutant `m`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillMatrix {
    /// Ordered list of test identifiers.
    pub tests: Vec<TestId>,
    /// Ordered list of mutant identifiers.
    pub mutants: Vec<MutantId>,
    /// Row-major matrix: `matrix[t][m]` is the outcome.
    pub matrix: Vec<Vec<TestOutcome>>,
}

impl KillMatrix {
    /// Create a new empty kill matrix.
    pub fn new(tests: Vec<TestId>, mutants: Vec<MutantId>) -> Self {
        let rows = tests.len();
        let cols = mutants.len();
        Self {
            tests,
            mutants,
            matrix: vec![vec![TestOutcome::NotRun; cols]; rows],
        }
    }

    /// Number of tests.
    pub fn num_tests(&self) -> usize {
        self.tests.len()
    }

    /// Number of mutants.
    pub fn num_mutants(&self) -> usize {
        self.mutants.len()
    }

    /// Set the outcome for test `t` against mutant `m`.
    pub fn set(&mut self, test_idx: usize, mutant_idx: usize, outcome: TestOutcome) {
        self.matrix[test_idx][mutant_idx] = outcome;
    }

    /// Get the outcome for test `t` against mutant `m`.
    pub fn get(&self, test_idx: usize, mutant_idx: usize) -> TestOutcome {
        self.matrix[test_idx][mutant_idx]
    }

    /// Get the index of a mutant by ID.
    pub fn mutant_index(&self, id: &MutantId) -> Option<usize> {
        self.mutants.iter().position(|m| m == id)
    }

    /// Get the index of a test by ID.
    pub fn test_index(&self, id: &TestId) -> Option<usize> {
        self.tests.iter().position(|t| t == id)
    }

    /// Returns the set of tests that kill a given mutant.
    pub fn killing_tests(&self, mutant_idx: usize) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();
        for (t, row) in self.matrix.iter().enumerate() {
            if row[mutant_idx].is_kill() {
                result.insert(t);
            }
        }
        result
    }

    /// Returns the set of mutants killed by a given test.
    pub fn killed_mutants(&self, test_idx: usize) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();
        for (m, outcome) in self.matrix[test_idx].iter().enumerate() {
            if outcome.is_kill() {
                result.insert(m);
            }
        }
        result
    }

    /// Returns true if the mutant at `mutant_idx` is killed by any test.
    pub fn is_killed(&self, mutant_idx: usize) -> bool {
        self.matrix.iter().any(|row| row[mutant_idx].is_kill())
    }

    /// Returns the kill set for each mutant (set of test indices that kill it).
    pub fn kill_sets(&self) -> Vec<BTreeSet<usize>> {
        (0..self.num_mutants())
            .map(|m| self.killing_tests(m))
            .collect()
    }

    /// Returns the set of mutant indices that are killed by at least one test.
    pub fn killed_set(&self) -> BTreeSet<usize> {
        (0..self.num_mutants())
            .filter(|&m| self.is_killed(m))
            .collect()
    }

    /// Returns the set of mutant indices that survived all tests.
    pub fn surviving_set(&self) -> BTreeSet<usize> {
        (0..self.num_mutants())
            .filter(|&m| !self.is_killed(m))
            .collect()
    }

    /// Returns a kill matrix restricted to the given mutant indices.
    pub fn restrict_mutants(&self, indices: &BTreeSet<usize>) -> KillMatrix {
        let new_mutants: Vec<MutantId> = indices.iter().map(|&i| self.mutants[i].clone()).collect();
        let new_matrix: Vec<Vec<TestOutcome>> = self
            .matrix
            .iter()
            .map(|row| indices.iter().map(|&i| row[i]).collect())
            .collect();
        KillMatrix {
            tests: self.tests.clone(),
            mutants: new_mutants,
            matrix: new_matrix,
        }
    }

    /// Returns a kill matrix restricted to the given test indices.
    pub fn restrict_tests(&self, indices: &BTreeSet<usize>) -> KillMatrix {
        let new_tests: Vec<TestId> = indices.iter().map(|&i| self.tests[i].clone()).collect();
        let new_matrix: Vec<Vec<TestOutcome>> =
            indices.iter().map(|&i| self.matrix[i].clone()).collect();
        KillMatrix {
            tests: new_tests,
            mutants: self.mutants.clone(),
            matrix: new_matrix,
        }
    }

    /// Create a KillMatrix from a boolean kill map for convenience.
    pub fn from_bool_matrix(
        tests: Vec<TestId>,
        mutants: Vec<MutantId>,
        kills: &[Vec<bool>],
    ) -> Self {
        let matrix = kills
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&k| {
                        if k {
                            TestOutcome::Killed
                        } else {
                            TestOutcome::Survived
                        }
                    })
                    .collect()
            })
            .collect();
        KillMatrix {
            tests,
            mutants,
            matrix,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Formula (from shared-types)
// ────────────────────────────────────────────────────────────────────────────

/// Logical formula in QF-LIA (quantifier-free linear integer arithmetic).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Formula {
    True,
    False,
    Predicate(Predicate),
    Not(Box<Formula>),
    And(Vec<Formula>),
    Or(Vec<Formula>),
    Implies(Box<Formula>, Box<Formula>),
}

impl Formula {
    pub fn and(conjuncts: Vec<Formula>) -> Self {
        let mut flat = Vec::new();
        for c in conjuncts {
            match c {
                Formula::True => {}
                Formula::And(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            Formula::True
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Formula::And(flat)
        }
    }

    pub fn or(disjuncts: Vec<Formula>) -> Self {
        let mut flat = Vec::new();
        for d in disjuncts {
            match d {
                Formula::False => {}
                Formula::Or(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.is_empty() {
            Formula::False
        } else if flat.len() == 1 {
            flat.into_iter().next().unwrap()
        } else {
            Formula::Or(flat)
        }
    }

    pub fn not(inner: Formula) -> Self {
        match inner {
            Formula::True => Formula::False,
            Formula::False => Formula::True,
            Formula::Not(f) => *f,
            other => Formula::Not(Box::new(other)),
        }
    }

    pub fn implies(lhs: Formula, rhs: Formula) -> Self {
        Formula::Implies(Box::new(lhs), Box::new(rhs))
    }

    /// Check if formula is trivially true.
    pub fn is_true(&self) -> bool {
        matches!(self, Formula::True)
    }

    /// Check if formula is trivially false.
    pub fn is_false(&self) -> bool {
        matches!(self, Formula::False)
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "true"),
            Formula::False => write!(f, "false"),
            Formula::Predicate(p) => write!(f, "{}", p),
            Formula::Not(inner) => write!(f, "¬({})", inner),
            Formula::And(parts) => {
                let strs: Vec<String> = parts.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", strs.join(" ∧ "))
            }
            Formula::Or(parts) => {
                let strs: Vec<String> = parts.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", strs.join(" ∨ "))
            }
            Formula::Implies(lhs, rhs) => write!(f, "({} → {})", lhs, rhs),
        }
    }
}

/// A comparison predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Predicate {
    pub op: RelOp,
    pub lhs: Term,
    pub rhs: Term,
}

impl Predicate {
    pub fn new(op: RelOp, lhs: Term, rhs: Term) -> Self {
        Self { op, lhs, rhs }
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.lhs, self.op, self.rhs)
    }
}

/// Relational operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for RelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => write!(f, "="),
            Self::Ne => write!(f, "≠"),
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "≤"),
            Self::Gt => write!(f, ">"),
            Self::Ge => write!(f, "≥"),
        }
    }
}

/// Arithmetic terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Var(String),
    Const(i64),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Mul(i64, Box<Term>),
    Neg(Box<Term>),
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    pub fn constant(v: i64) -> Self {
        Term::Const(v)
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(name) => write!(f, "{}", name),
            Term::Const(v) => write!(f, "{}", v),
            Term::Add(l, r) => write!(f, "({} + {})", l, r),
            Term::Sub(l, r) => write!(f, "({} - {})", l, r),
            Term::Mul(c, t) => write!(f, "({} * {})", c, t),
            Term::Neg(t) => write!(f, "(-{})", t),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SMT solver interface (from smt-solver)
// ────────────────────────────────────────────────────────────────────────────

/// Result of an SMT check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable, with an optional model (variable → value).
    Sat(BTreeMap<String, i64>),
    /// Unsatisfiable.
    Unsat,
    /// Solver could not determine.
    Unknown(String),
}

impl SolverResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, Self::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, Self::Unsat)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    pub fn model(&self) -> Option<&BTreeMap<String, i64>> {
        match self {
            Self::Sat(m) => Some(m),
            _ => None,
        }
    }
}

/// Trait abstracting an SMT solver.
pub trait SmtSolver: Send + Sync {
    /// Check satisfiability of a formula.
    fn check_sat(&self, formula: &Formula) -> SolverResult;

    /// Check if `premise` implies `conclusion`.
    fn check_implies(&self, premise: &Formula, conclusion: &Formula) -> SolverResult {
        // premise |= conclusion iff (premise ∧ ¬conclusion) is UNSAT
        let negated = Formula::and(vec![premise.clone(), Formula::not(conclusion.clone())]);
        self.check_sat(&negated)
    }

    /// Check if two formulas are equivalent.
    fn check_equivalent(&self, f1: &Formula, f2: &Formula) -> SolverResult {
        // f1 ≡ f2 iff (f1 XOR f2) is UNSAT
        let xor = Formula::or(vec![
            Formula::and(vec![f1.clone(), Formula::not(f2.clone())]),
            Formula::and(vec![Formula::not(f1.clone()), f2.clone()]),
        ]);
        self.check_sat(&xor)
    }

    /// Get the solver name.
    fn name(&self) -> &str;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn SmtSolver>;
}

/// A trivial solver used for testing. Always returns a configurable result.
#[derive(Debug, Clone)]
pub struct TrivialSolver {
    pub default_result: TrivialSolverMode,
}

#[derive(Debug, Clone)]
pub enum TrivialSolverMode {
    AlwaysSat,
    AlwaysUnsat,
    AlwaysUnknown,
}

impl SmtSolver for TrivialSolver {
    fn check_sat(&self, _formula: &Formula) -> SolverResult {
        match &self.default_result {
            TrivialSolverMode::AlwaysSat => SolverResult::Sat(BTreeMap::new()),
            TrivialSolverMode::AlwaysUnsat => SolverResult::Unsat,
            TrivialSolverMode::AlwaysUnknown => SolverResult::Unknown("trivial solver".to_string()),
        }
    }

    fn name(&self) -> &str {
        "TrivialSolver"
    }

    fn clone_box(&self) -> Box<dyn SmtSolver> {
        Box::new(self.clone())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Coverage errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during coverage analysis.
#[derive(Debug, thiserror::Error)]
pub enum CoverageError {
    #[error("mutant not found: {0}")]
    MutantNotFound(MutantId),

    #[error("test not found: {0}")]
    TestNotFound(TestId),

    #[error("kill matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("solver error: {0}")]
    SolverError(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("computation error: {0}")]
    ComputationError(String),

    #[error("empty input: {0}")]
    EmptyInput(String),

    #[error("index out of bounds: {index} >= {bound}")]
    IndexOutOfBounds { index: usize, bound: usize },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, CoverageError>;

// ────────────────────────────────────────────────────────────────────────────
// Utilities
// ────────────────────────────────────────────────────────────────────────────

/// Helper to create mutant descriptors for testing.
#[cfg(test)]
pub(crate) fn make_test_mutant(id: &str, op: MutationOperator) -> MutantDescriptor {
    MutantDescriptor::new(
        MutantId::new(id),
        op,
        MutationSite::new("test.c", "test_func", 1, 1),
        "original",
        "replacement",
    )
}

/// Helper to create a small kill matrix for testing.
#[cfg(test)]
pub(crate) fn make_test_kill_matrix(
    num_tests: usize,
    num_mutants: usize,
    kills: &[(usize, usize)],
) -> KillMatrix {
    let tests: Vec<TestId> = (0..num_tests)
        .map(|i| TestId::new(format!("t{}", i)))
        .collect();
    let mutants: Vec<MutantId> = (0..num_mutants)
        .map(|i| MutantId::new(format!("m{}", i)))
        .collect();
    let mut km = KillMatrix::new(tests, mutants);
    for &(t, m) in kills {
        km.set(t, m, TestOutcome::Killed);
    }
    km
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutant_id_display() {
        let id = MutantId::new("mut_001");
        assert_eq!(id.to_string(), "mut_001");
        assert_eq!(id.as_str(), "mut_001");
    }

    #[test]
    fn test_test_id_display() {
        let id = TestId::new("test_add");
        assert_eq!(id.to_string(), "test_add");
    }

    #[test]
    fn test_mutation_operator_all() {
        let all = MutationOperator::all();
        assert_eq!(all.len(), 13);
        assert!(all.contains(&MutationOperator::AOR));
        assert!(all.contains(&MutationOperator::BOMB));
    }

    #[test]
    fn test_kill_matrix_basic() {
        let km = make_test_kill_matrix(3, 4, &[(0, 0), (0, 1), (1, 1), (2, 2)]);
        assert_eq!(km.num_tests(), 3);
        assert_eq!(km.num_mutants(), 4);
        assert!(km.is_killed(0));
        assert!(km.is_killed(1));
        assert!(km.is_killed(2));
        assert!(!km.is_killed(3));
    }

    #[test]
    fn test_kill_matrix_killing_tests() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (1, 0), (2, 1)]);
        let kt0 = km.killing_tests(0);
        assert_eq!(kt0.len(), 2);
        assert!(kt0.contains(&0));
        assert!(kt0.contains(&1));
        let kt1 = km.killing_tests(1);
        assert_eq!(kt1.len(), 1);
        assert!(kt1.contains(&2));
    }

    #[test]
    fn test_kill_matrix_killed_mutants() {
        let km = make_test_kill_matrix(3, 3, &[(0, 0), (0, 2), (1, 1)]);
        let km0 = km.killed_mutants(0);
        assert_eq!(km0.len(), 2);
        assert!(km0.contains(&0));
        assert!(km0.contains(&2));
    }

    #[test]
    fn test_kill_matrix_killed_and_surviving_sets() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (1, 2)]);
        let killed = km.killed_set();
        assert_eq!(killed, BTreeSet::from([0, 2]));
        let surviving = km.surviving_set();
        assert_eq!(surviving, BTreeSet::from([1, 3]));
    }

    #[test]
    fn test_kill_matrix_restrict_mutants() {
        let km = make_test_kill_matrix(2, 4, &[(0, 0), (0, 2), (1, 1), (1, 3)]);
        let restricted = km.restrict_mutants(&BTreeSet::from([0, 2]));
        assert_eq!(restricted.num_mutants(), 2);
        assert_eq!(restricted.mutants[0], MutantId::new("m0"));
        assert_eq!(restricted.mutants[1], MutantId::new("m2"));
        assert!(restricted.get(0, 0).is_kill());
        assert!(restricted.get(0, 1).is_kill());
    }

    #[test]
    fn test_formula_smart_constructors() {
        let f = Formula::and(vec![Formula::True, Formula::True]);
        assert!(f.is_true());
        let f = Formula::or(vec![Formula::False, Formula::False]);
        assert!(f.is_false());
        let f = Formula::not(Formula::True);
        assert!(f.is_false());
        let f = Formula::not(Formula::not(Formula::True));
        assert!(f.is_true());
    }

    #[test]
    fn test_formula_and_flattening() {
        let inner = Formula::and(vec![
            Formula::Predicate(Predicate::new(RelOp::Eq, Term::var("x"), Term::Const(1))),
            Formula::Predicate(Predicate::new(RelOp::Eq, Term::var("y"), Term::Const(2))),
        ]);
        let outer = Formula::and(vec![
            inner,
            Formula::Predicate(Predicate::new(RelOp::Gt, Term::var("z"), Term::Const(0))),
        ]);
        match outer {
            Formula::And(parts) => assert_eq!(parts.len(), 3),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn test_solver_result_helpers() {
        let sat = SolverResult::Sat(BTreeMap::from([("x".into(), 42)]));
        assert!(sat.is_sat());
        assert!(!sat.is_unsat());
        assert!(sat.model().is_some());

        let unsat = SolverResult::Unsat;
        assert!(unsat.is_unsat());
        assert!(unsat.model().is_none());

        let unk = SolverResult::Unknown("timeout".into());
        assert!(unk.is_unknown());
    }

    #[test]
    fn test_trivial_solver() {
        let solver = TrivialSolver {
            default_result: TrivialSolverMode::AlwaysUnsat,
        };
        let result = solver.check_sat(&Formula::True);
        assert!(result.is_unsat());

        let solver = TrivialSolver {
            default_result: TrivialSolverMode::AlwaysSat,
        };
        let result = solver.check_sat(&Formula::False);
        assert!(result.is_sat());
    }

    #[test]
    fn test_mutant_descriptor_display() {
        let desc = MutantDescriptor::new(
            MutantId::new("m1"),
            MutationOperator::AOR,
            MutationSite::new("main.c", "add", 10, 5),
            "+",
            "-",
        );
        assert!(desc.to_string().contains("AOR"));
        assert!(desc.to_string().contains("+"));
        assert!(desc.to_string().contains("-"));
    }

    #[test]
    fn test_mutant_status() {
        assert!(MutantStatus::Pending.is_pending());
        assert!(MutantStatus::Alive.is_alive());
        assert!(MutantStatus::Equivalent.is_equivalent());
        let killed = MutantStatus::Killed(KillInfo::new(TestId::new("t1")));
        assert!(killed.is_killed());
    }

    #[test]
    fn test_from_bool_matrix() {
        let tests = vec![TestId::new("t0"), TestId::new("t1")];
        let mutants = vec![MutantId::new("m0"), MutantId::new("m1")];
        let kills = vec![vec![true, false], vec![false, true]];
        let km = KillMatrix::from_bool_matrix(tests, mutants, &kills);
        assert!(km.get(0, 0).is_kill());
        assert!(!km.get(0, 1).is_kill());
        assert!(!km.get(1, 0).is_kill());
        assert!(km.get(1, 1).is_kill());
    }

    #[test]
    fn test_kill_matrix_kill_sets() {
        let km = make_test_kill_matrix(3, 2, &[(0, 0), (1, 0), (2, 0), (2, 1)]);
        let sets = km.kill_sets();
        assert_eq!(sets[0], BTreeSet::from([0, 1, 2]));
        assert_eq!(sets[1], BTreeSet::from([2]));
    }
}
